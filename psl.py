"""
PSL sign recognition: MediaPipe Holistic landmarks -> small GRU.
Handles letters (static) and words (motion) with ONE model.

pip install mediapipe==0.10.21 opencv-python "numpy<2" torch fastapi "uvicorn[standard]"

python psl.py record --label alif --signer m     # SPACE = record a clip, q = quit
python psl.py train                              # trains on data/, saves psl_model.pt
python psl.py live                               # webcam test (box + skeleton + big text)
python psl.py api                                # web app + API on http://localhost:8000

Data layout: data/<label>/<label>__<signer>__<time>.npy
"""
import sys, os, glob, time, json, argparse
from collections import deque
import numpy as np

SEQ = 30          # frames per clip
F = 140           # features per frame (see features())
DATA = "data"
MODEL = "psl_model.pt"


# ---------------- features ----------------
def _hand(lm):
    """21 landmarks -> wrist-relative, scale-normalised shape (63) + wrist xy."""
    if lm is None:
        return np.zeros(63, np.float32), None
    p = np.array([[l.x, l.y, l.z] for l in lm.landmark], np.float32)
    wrist = p[0, :2].copy()
    q = p - p[0]
    s = np.linalg.norm(q[9, :2]) + 1e-6          # wrist -> middle-finger base
    return (q / s).flatten(), wrist


def features(res):
    """MediaPipe Holistic result -> 140-dim vector.
    [left hand shape 63 | right hand shape 63 | left wrist vs body 2 | right wrist vs body 2 | body 10]"""
    if res.pose_landmarks is not None:
        P = np.array([[l.x, l.y] for l in res.pose_landmarks.landmark], np.float32)
        c = (P[11] + P[12]) / 2
        w = np.linalg.norm(P[11] - P[12]) + 1e-6
        body = ((P[[0, 11, 12, 13, 14]] - c) / w).flatten()   # nose, shoulders, elbows
    else:
        c, w, body = np.array([.5, .5], np.float32), 0.3, np.zeros(10, np.float32)
    lh, lw = _hand(res.left_hand_landmarks)
    rh, rw = _hand(res.right_hand_landmarks)
    lpos = (lw - c) / w if lw is not None else np.zeros(2, np.float32)
    rpos = (rw - c) / w if rw is not None else np.zeros(2, np.float32)
    return np.concatenate([lh, rh, lpos, rpos, body]).astype(np.float32)


def hands_present(res):
    return res.left_hand_landmarks is not None or res.right_hand_landmarks is not None


def holistic():
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)


# ---------------- drawing (box + skeleton + readable text) ----------------
def put_text(img, text, org, scale=1.8, color=(0, 255, 0)):
    """Text with a thick black outline so it is readable on any background."""
    import cv2
    f = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, text, org, f, scale, (0, 0, 0), 9, cv2.LINE_AA)
    cv2.putText(img, text, org, f, scale, color, 3, cv2.LINE_AA)


def draw_overlay(frame, res, text="", ok=False):
    """Draws a box and skeleton on each hand, and a big banner with the prediction."""
    import cv2, mediapipe as mp
    h, w = frame.shape[:2]
    mpd, mph = mp.solutions.drawing_utils, mp.solutions.holistic
    box_col = (0, 255, 0) if ok else (0, 215, 255)
    for lm in (res.left_hand_landmarks, res.right_hand_landmarks):
        if lm is None:
            continue
        mpd.draw_landmarks(
            frame, lm, mph.HAND_CONNECTIONS,
            mpd.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4),
            mpd.DrawingSpec(color=(255, 255, 255), thickness=3))
        xs = [l.x for l in lm.landmark]
        ys = [l.y for l in lm.landmark]
        x0, y0 = int(max(min(xs) - .03, 0) * w), int(max(min(ys) - .03, 0) * h)
        x1, y1 = int(min(max(xs) + .03, 1) * w), int(min(max(ys) + .03, 1) * h)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), 7)
        cv2.rectangle(frame, (x0, y0), (x1, y1), box_col, 3)
    if text:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.8, 3)
        cv2.rectangle(frame, (0, 0), (tw + 40, th + 40), (0, 0, 0), -1)
        put_text(frame, text, (20, th + 20), 1.8, (0, 255, 0) if ok else (0, 215, 255))
    return frame


class Smoother:
    """Majority vote over the last few predictions so the label stops flickering."""
    def __init__(self, n=7, thr=0.6):
        self.v, self.thr = deque(maxlen=n), thr

    def update(self, label, p):
        self.v.append(label if p >= self.thr else None)
        counts = {}
        for x in self.v:
            if x:
                counts[x] = counts.get(x, 0) + 1
        if not counts:
            return ""
        best = max(counts, key=counts.get)
        return best if counts[best] >= max(3, len(self.v) // 2) else ""

    def clear(self):
        self.v.clear()


# ---------------- model ----------------
def build_model(n_classes):
    import torch, torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(F * 2, 128, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))

        def forward(self, x):                        # x: (B, T, F)
            v = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], 1)  # velocity
            out, _ = self.gru(torch.cat([x, v], -1))
            return self.head(out[:, -1])

    return Net()


# ---------------- record ----------------
def record(label, signer):
    import cv2
    os.makedirs(f"{DATA}/{label}", exist_ok=True)
    cap, hol = cv2.VideoCapture(0), holistic()
    n = len(glob.glob(f"{DATA}/{label}/*.npy"))
    print(f"[{label}] {n} clips so far. SPACE=record, q=quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw_overlay(frame, res, f"{label}  clips: {n}", ok=hands_present(res))
        cv2.imshow("record  (SPACE=rec, q=quit)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == 32:
            seq = []
            while len(seq) < SEQ:
                ok, frame = cap.read()
                res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                seq.append(features(res))
                draw_overlay(frame, res, "REC", ok=False)
                cv2.imshow("record  (SPACE=rec, q=quit)", frame)
                cv2.waitKey(1)
            np.save(f"{DATA}/{label}/{label}__{signer}__{int(time.time())}.npy", np.array(seq))
            n += 1
            print("saved", n)
    cap.release()
    cv2.destroyAllWindows()


# ---------------- train ----------------
def augment(x):
    """x: (T, F). Small in-plane rotation of each hand, scale + noise."""
    x = x.copy()
    a = np.deg2rad(np.random.uniform(-25, 25))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    for off in (0, 63):
        h = x[:, off:off + 63].reshape(-1, 21, 3)
        h[:, :, :2] = h[:, :, :2] @ R.T
        x[:, off:off + 63] = h.reshape(-1, 63)
    x *= np.random.uniform(0.95, 1.05)
    x += np.random.normal(0, 0.01, x.shape).astype(np.float32)
    return x


def train(epochs=80):
    import torch, torch.nn as nn
    files = sorted(glob.glob(f"{DATA}/*/*.npy"))
    labels = sorted({f.split(os.sep)[-2] for f in files})
    X = [np.load(f) for f in files]
    y = [labels.index(f.split(os.sep)[-2]) for f in files]
    signers = [os.path.basename(f).split("__")[1] for f in files]
    json.dump(labels, open("labels.json", "w"))

    uniq = sorted(set(signers))
    if len(uniq) > 1:                                  # split BY SIGNER
        val_s = uniq[-1]
        tr = [i for i, s in enumerate(signers) if s != val_s]
        va = [i for i, s in enumerate(signers) if s == val_s]
        print(f"validating on signer '{val_s}'")
    else:
        idx = np.random.permutation(len(X)); k = int(len(X) * .8)
        tr, va = list(idx[:k]), list(idx[k:])
        print("WARNING: one signer only -> val accuracy will look better than reality")

    model = build_model(len(labels))
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-3)
    best = 0
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(tr)
        for b in range(0, len(order), 32):
            ids = order[b:b + 32]
            xb = torch.tensor(np.stack([augment(X[i]) for i in ids]))
            yb = torch.tensor([y[i] for i in ids])
            loss = nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(np.stack([X[i] for i in va]))).argmax(1).numpy()
        acc = float((pred == np.array([y[i] for i in va])).mean())
        if acc >= best:                                # keep BEST, not last
            best = acc
            torch.save({"state": model.state_dict(), "labels": labels}, MODEL)
        if ep % 5 == 0:
            print(f"epoch {ep:3d}  loss {loss.item():.3f}  val acc {acc:.3f}  best {best:.3f}")
    print("best val acc", best, "->", MODEL)


# ---------------- inference ----------------
def load():
    import torch
    ck = torch.load(MODEL, map_location="cpu")
    m = build_model(len(ck["labels"]))
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck["labels"]


def predict_top(m, labels, seq, k=3):
    """Returns [[label, prob], ...] best first."""
    import torch
    with torch.no_grad():
        p = torch.softmax(m(torch.tensor(np.array(seq, np.float32))[None]), -1)[0]
    idx = p.argsort(descending=True)[:k]
    return [[labels[int(i)], round(float(p[int(i)]), 3)] for i in idx]


def predict(m, labels, seq):
    top = predict_top(m, labels, seq, 1)[0]
    return top[0], top[1]


def live():
    import cv2
    m, labels = load()
    cap, hol = cv2.VideoCapture(0), holistic()
    buf, sm, shown, ok_flag, n = deque(maxlen=SEQ), Smoother(), "", False, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buf.append(features(res)); n += 1
        if hands_present(res):
            if len(buf) == SEQ and n % 3 == 0:
                lab, p = predict(m, labels, buf)
                s = sm.update(lab, p)
                shown, ok_flag = (f"{s}  {int(p * 100)}%", True) if s else ("...", False)
        else:
            sm.clear(); shown, ok_flag = "show a sign", False
        draw_overlay(frame, res, shown, ok=ok_flag)
        cv2.imshow("live (q=quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# ---------------- web app + API ----------------
PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sync - PSL to text</title>
<style>
 body{margin:0;background:#0d1117;color:#fff;font-family:system-ui,sans-serif;text-align:center}
 h1{font-size:20px;margin:12px 0 6px}
 #wrap{position:relative;display:inline-block;width:100%;max-width:640px}
 video,canvas{width:100%;display:block;transform:scaleX(-1)}
 canvas{position:absolute;left:0;top:0}
 #sign{font-size:min(20vw,120px);font-weight:800;color:#39ff14;min-height:1.2em;line-height:1.2;
       text-shadow:-2px -2px 0 #000,2px -2px 0 #000,-2px 2px 0 #000,2px 2px 0 #000}
 #sign.low{color:#ffd400}
 #sub{font-size:20px;color:#ddd;min-height:28px}
 button{font-size:20px;padding:12px 28px;border:0;border-radius:10px;background:#39ff14;color:#000;font-weight:700;margin:10px}
</style></head><body>
<h1>Sync - Pakistan Sign Language</h1>
<div id="wrap"><video id="v" playsinline muted></video><canvas id="c"></canvas></div>
<div id="sign">...</div>
<div id="sub"></div>
<button id="go">Start camera</button>
<script>
const v=document.getElementById('v'), c=document.getElementById('c'), ctx=c.getContext('2d');
const off=document.createElement('canvas'), octx=off.getContext('2d');
const signEl=document.getElementById('sign'), subEl=document.getElementById('sub');
const CONN=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
let ws;
function send(){
  octx.drawImage(v,0,0,off.width,off.height);
  off.toBlob(b=>{ if(b && ws && ws.readyState===1) ws.send(b); },'image/jpeg',0.6);
}
function draw(d){
  ctx.clearRect(0,0,c.width,c.height);
  const ok = !!d.sign;
  (d.hands||[]).forEach(h=>{
    const [x0,y0,x1,y1]=h.box;
    ctx.lineWidth=6; ctx.strokeStyle='#000';
    ctx.strokeRect(x0*c.width,y0*c.height,(x1-x0)*c.width,(y1-y0)*c.height);
    ctx.lineWidth=3; ctx.strokeStyle= ok ? '#39ff14' : '#ffd400';
    ctx.strokeRect(x0*c.width,y0*c.height,(x1-x0)*c.width,(y1-y0)*c.height);
    ctx.lineWidth=3; ctx.strokeStyle='#fff';
    CONN.forEach(([a,b])=>{ctx.beginPath();ctx.moveTo(h.pts[a][0]*c.width,h.pts[a][1]*c.height);
                           ctx.lineTo(h.pts[b][0]*c.width,h.pts[b][1]*c.height);ctx.stroke();});
    ctx.fillStyle='#f00';
    h.pts.forEach(p=>{ctx.beginPath();ctx.arc(p[0]*c.width,p[1]*c.height,4,0,7);ctx.fill();});
  });
  if(!(d.hands||[]).length){ signEl.textContent='show a sign'; signEl.className='low'; subEl.textContent=''; return; }
  signEl.textContent = d.sign || '...';
  signEl.className = d.sign ? '' : 'low';
  subEl.textContent = (d.top||[]).map(t=>t[0]+' '+Math.round(t[1]*100)+'%').join('   ');
}
document.getElementById('go').onclick = async ()=>{
  const s = await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:640,height:480},audio:false});
  v.srcObject = s; await v.play();
  c.width=v.videoWidth; c.height=v.videoHeight;
  off.width=320; off.height=Math.round(320*v.videoHeight/v.videoWidth);
  ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
  ws.onopen = send;
  ws.onmessage = e=>{ draw(JSON.parse(e.data)); send(); };   // next frame only after reply
  document.getElementById('go').style.display='none';
};
</script></body></html>"""


def api():
    """Web app at / (camera in browser) + WebSocket /ws + POST /predict.
    The server runs the SAME MediaPipe Holistic + model as live/train,
    so web results match what you trained on."""
    import uvicorn, cv2
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    from starlette.concurrency import run_in_threadpool

    m, labels = load()
    app = FastAPI()

    def step(state, data):
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {}
        res = state["hol"].process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        state["buf"].append(features(res))
        hands = []
        for lm in (res.left_hand_landmarks, res.right_hand_landmarks):
            if lm is None:
                continue
            pts = [[round(l.x, 3), round(l.y, 3)] for l in lm.landmark]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            box = [max(min(xs) - .03, 0), max(min(ys) - .03, 0),
                   min(max(xs) + .03, 1), min(max(ys) + .03, 1)]
            hands.append({"pts": pts, "box": [round(b, 3) for b in box]})
        sign, top = "", []
        if hands_present(res):
            state["nohand"] = 0
            if len(state["buf"]) == SEQ:
                top = predict_top(m, labels, state["buf"])
                sign = state["sm"].update(top[0][0], top[0][1])
        else:
            state["nohand"] += 1
            if state["nohand"] > 10:
                state["sm"].clear()
        return {"hands": hands, "sign": sign, "top": top}

    @app.get("/", response_class=HTMLResponse)
    def home():
        return PAGE

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        state = {"hol": holistic(), "buf": deque(maxlen=SEQ), "sm": Smoother(), "nohand": 0}
        try:
            while True:
                data = await sock.receive_bytes()
                await sock.send_json(await run_in_threadpool(step, state, data))
        except WebSocketDisconnect:
            pass
        finally:
            state["hol"].close()

    class Req(BaseModel):
        seq: list[list[float]]          # 30 frames x 140 features

    @app.post("/predict")
    def _predict(r: Req):
        lab, p = predict(m, labels, r.seq)
        return {"sign": lab, "confidence": p}

    print("Open http://localhost:8000  (for phones use an HTTPS tunnel, see README)")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "train", "live", "api"])
    ap.add_argument("--label"); ap.add_argument("--signer", default="m")
    a = ap.parse_args()
    {"record": lambda: record(a.label, a.signer), "train": train, "live": live, "api": api}[a.mode]()
