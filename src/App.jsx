import React, { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";

// --- Tiny helper: seeded RNG ---
function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- Toddler-oriented word list (>=250). Slice to 250 max in UI.
// You can paste your own list; the app will respect the 250 cap.
const TODDLER_WORDS = [
  // family & people
  "mommy","daddy","mama","papa","baby","brother","sister","grandma","grandpa","aunt","uncle","cousin","friend","teacher","doctor","neighbor",
  // pronouns & possessives
  "I","me","you","we","it","he","she","they","my","mine","your","yours","our","ours","his","her","their","theirs","self",
  // common animals
  "dog","puppy","cat","kitten","bird","duck","chicken","cow","horse","sheep","goat","pig","bunny","rabbit","mouse","monkey","lion","tiger","bear","wolf","fox","deer","elephant","giraffe","zebra","kangaroo","whale","dolphin","fish","frog","turtle","snake","bee","ant","spider","bug",
  // foods & kitchen
  "milk","water","juice","apple","banana","orange","grape","pear","peach","strawberry","blueberry","melon","carrot","broccoli","pea","corn","potato","tomato","egg","bread","toast","cheese","yogurt","cracker","cookie","cake","candy","cereal","pasta","rice","soup","pizza","sandwich","chicken","burger","butter","jam","honey","salt","pepper","cup","bottle","spoon","fork","knife","plate","bowl","napkin",
  // house & objects
  "table","chair","desk","bed","blanket","quilt","pillow","lamp","light","door","window","floor","wall","ceiling","clock","tv","remote","phone","book","page","pen","pencil","paper","crayon","marker","sticker","bag","backpack",
  // toys & vehicles
  "ball","blocks","truck","car","bus","train","boat","plane","bike","trike","scooter","stroller","doll","robot","puzzle","drum","guitar","lego",
  // body parts
  "head","face","hair","eye","ear","nose","mouth","teeth","tongue","neck","shoulder","arm","elbow","hand","finger","thumb","belly","back","leg","knee","foot","toe","heart",
  // clothes
  "shirt","t-shirt","pants","jeans","shorts","dress","skirt","coat","jacket","hoodie","hat","cap","scarf","glove","mitten","sock","shoe","boot","diaper",
  // places & nature
  "home","house","room","kitchen","bathroom","bedroom","yard","garden","park","playground","school","store","farm","forest","beach","road","street","bridge","sidewalk","trail","river","lake","pond","mountain","hill","sand","snow","rain","cloud","sun","moon","star",
  // verbs (actions)
  "go","come","stop","wait","look","see","watch","listen","hear","touch","hold","carry","open","close","turn","push","pull","put","take","give","help","make","build","draw","paint","color","cut","glue","read","write","sing","play","run","walk","crawl","jump","climb","slide","hide","seek","throw","catch","kick","dance","swim","ride","sleep","wake","nap","eat","drink","chew","lick","bite","wash","brush","comb","pee","poop","change","clean","cook","bake","drive","fly","hug","kiss","laugh","cry","smile","share",
  // adjectives & states
  "big","small","little","tiny","huge","tall","short","long","new","old","hot","cold","warm","cool","wet","dry","soft","hard","loud","quiet","noisy","fast","slow","happy","sad","angry","mad","silly","funny","nice","kind","gentle","pretty","beautiful","dirty","clean","empty","full","open","closed","broken","fixed","yummy","yucky","tasty",
  // colors
  "red","blue","green","yellow","orange","purple","pink","brown","black","white","gray",
  // positions & prepositions
  "in","on","under","over","up","down","off","out","inside","outside","near","far","here","there","behind","front","between","next","around","through","beside","above","below",
  // quantities & time
  "one","two","three","four","five","six","seven","eight","nine","ten","more","most","some","all","none","any","another","again","first","last","now","later","today","tonight","tomorrow","yesterday","soon",
  // social words
  "and","or","not","yes","no","please","thank","thanks","welcome","sorry","hello","hi","bye","goodbye","good","morning","night","goodnight","okay","ok"
];

// color: unique per node via HSL
function colorByIndex(i) {
  const hue = (i * 137.508) % 360; // golden angle for separation
  return `hsl(${hue}, 75%, 55%)`;
}

// cosine similarity
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / Math.sqrt(na * nb);
}

// generate embeddings; we add light structure so similar categories drift closer
function makeEmbeddings(words, dim, seed) {
  const rnd = mulberry32(seed);
  const base = (label) => {
    // crude categories by heuristics
    if (/(mom|dad|grand|aunt|uncle|he|she|they|you|I)/i.test(label)) return 0;
    if (/(dog|cat|puppy|kitten|cow|horse|sheep|goat|pig|bear|wolf|fox|deer|lion|tiger|elephant|giraffe|zebra|duck|chicken|fish|frog|turtle)/i.test(label)) return 1;
    if (/(go|come|run|walk|jump|climb|play|sleep|eat|drink|open|close|push|pull|throw|catch|give|help|make|build|draw|read)/i.test(label)) return 2;
    if (/(big|small|tiny|huge|hot|cold|red|blue|green|happy|sad|yummy|yucky|soft|hard)/i.test(label)) return 3;
    if (/(in|on|under|over|up|down|out|inside|outside|near|far)/i.test(label)) return 4;
    if (/(the|and|or|yes|no|please|thank|hello|hi|bye|okay|ok)/i.test(label)) return 5;
    if (/(milk|water|juice|apple|banana|bread|cheese|cookie|pizza|sandwich|cup|spoon|bowl)/i.test(label)) return 6;
    return 7;
  };
  const centers = Array.from({ length: 8 }, () =>
    Array.from({ length: dim }, () => rnd() * 2 - 1)
  );
  // L2 normalize centers
  centers.forEach((c) => {
    const n = Math.sqrt(c.reduce((s, v) => s + v * v, 0)) || 1;
    for (let i = 0; i < c.length; i++) c[i] /= n;
  });

  const embs = {};
  for (const w of words) {
    const c = centers[base(w)];
    const v = Array.from({ length: dim }, () => (rnd() * 2 - 1) * 0.35);
    // mix with center
    const out = v.map((vi, i) => vi + c[i] * 0.8 + (rnd() * 2 - 1) * 0.05);
    // normalize
    const n = Math.sqrt(out.reduce((s, x) => s + x * x, 0)) || 1;
    embs[w] = out.map((x) => x / n);
  }
  return embs;
}

// Build graph from embeddings and threshold
function buildGraph(words, embs, threshold) {
  const nodes = words.map((w, i) => ({ id: w, idx: i }));
  const links = [];
  for (let i = 0; i < words.length; i++) {
    for (let j = i + 1; j < words.length; j++) {
      const a = words[i], b = words[j];
      const sim = cosine(embs[a], embs[b]);
      if (sim >= threshold) links.push({ source: a, target: b, value: sim });
    }
  }
  return { nodes, links };
}

// Compute largest connected component size
function largestComponentSize(graph) {
  const adj = new Map();
  graph.nodes.forEach((n) => adj.set(n.id, []));
  graph.links.forEach((l) => {
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    adj.get(s).push(t);
    adj.get(t).push(s);
  });
  const seen = new Set();
  let best = 0;
  for (const n of graph.nodes) {
    if (seen.has(n.id)) continue;
    let size = 0;
    const q = [n.id];
    seen.add(n.id);
    while (q.length) {
      const cur = q.shift();
      size++;
      for (const nb of adj.get(cur) || []) {
        if (!seen.has(nb)) {
          seen.add(nb);
          q.push(nb);
        }
      }
    }
    best = Math.max(best, size);
  }
  return best;
}

export default function App() {
  const fgRef = useRef(null);

  // Controls
  const [vocabN, setVocabN] = useState(120);
  const [embedDim, setEmbedDim] = useState(16);
  const [seed, setSeed] = useState(42);
  const [threshold, setThreshold] = useState(0.32);
  const [autoRotate, setAutoRotate] = useState(true);

  // Derived word list (trim to N, cap at 250)
  const WORD_CAP = 250;
  const ALL_WORDS = useMemo(() => TODDLER_WORDS.slice(0, Math.max(WORD_CAP, TODDLER_WORDS.length)), []);
  const sliderMax = Math.min(WORD_CAP, ALL_WORDS.length);
  const words = useMemo(() => ALL_WORDS.slice(0, Math.min(vocabN, sliderMax)), [vocabN, sliderMax]);

  const embs = useMemo(() => makeEmbeddings(words, embedDim, seed), [words, embedDim, seed]);
  const graph = useMemo(() => buildGraph(words, embs, threshold), [words, embs, threshold]);
  const giant = useMemo(() => largestComponentSize(graph), [graph]);

  // Percolation indicator (simple heuristic: giant covers >= 60%)
  const giantRatio = words.length ? giant / words.length : 0;
  const isPercolating = giantRatio >= 0.6;

  // Fit to screen on first load and when graph changes
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    const t = setTimeout(() => {
      try {
        fg.zoomToFit(800, 50, () => true);
      } catch {}
    }, 80);
    return () => clearTimeout(t);
  }, [graph]);

  // Use OrbitControls auto-rotate (robust)
  useEffect(() => {
    const fg = fgRef.current;
    const controls = fg?.controls?.();
    if (!controls) return;
    controls.autoRotate = !!autoRotate;
    controls.autoRotateSpeed = 0.6; // gentle
    return () => { controls.autoRotate = false; };
  }, [autoRotate]);

  // Make nodes visually closer: short link distance + softer charge
  const forceInit = (fg) => {
    fg.d3VelocityDecay(0.2);
    fg.d3Force("link")?.distance(22).strength(0.6);
    fg.d3Force("charge")?.strength(-18);
    fg.d3Force("center")?.strength(0.9);
  };

  // HUD stats
  const edgeCount = graph.links.length;
  const density = words.length > 1 ? (edgeCount * 2) / (words.length * (words.length - 1)) : 0;

  return (
    <div className="w-full h-screen grid grid-cols-12 bg-slate-900 text-slate-100">
      {/* Left controls */}
      <style>{`
        @keyframes pulseFlash { 0%{background-color:rgba(21,128,61,.15)} 50%{background-color:rgba(21,128,61,.4)} 100%{background-color:rgba(21,128,61,.15)} }
      `}</style>
      <div className="col-span-3 p-4 space-y-4 overflow-y-auto border-r border-slate-700">
        <h1 className="text-xl font-semibold">Word Sentence Simulator</h1>
        <div className="text-xs text-slate-400">Max vocabulary: 250 words</div>

        <HelpRow label="Vocabulary size (N)" desc="How many words are included from a toddler’s common lexicon. More words usually means more possible links and a larger connected cluster.">
          <input type="range" min={10} max={sliderMax} value={vocabN} onChange={(e)=>setVocabN(parseInt(e.target.value))} className="w-full" />
          <div className="text-sm">{vocabN} / {sliderMax}</div>
        </HelpRow>

        <HelpRow label="Embedding dim" desc="How many features each word has in the semantic space. Higher dimension lets the model express more nuanced similarities (often helping real relationships stand out), but too high with few words can add noise.">
          <input type="range" min={4} max={64} step={1} value={embedDim} onChange={(e)=>setEmbedDim(parseInt(e.target.value))} className="w-full" />
          <div className="text-sm">{embedDim}D</div>
        </HelpRow>

        <HelpRow label="Similarity threshold" desc="Minimum cosine similarity required to draw an edge between two words. Lower threshold = more edges; higher = only very close neighbors connect.">
          <input type="range" min={0.1} max={0.9} step={0.01} value={threshold} onChange={(e)=>setThreshold(parseFloat(e.target.value))} className="w-full" />
          <div className="text-sm">{threshold.toFixed(2)}</div>
        </HelpRow>

        <HelpRow label="Seed" desc="Sets the pseudo‑random generator so you can reproduce a layout/embedding. Change it to explore alternative structures with the same settings.">
          <input type="number" className="w-full bg-slate-800 rounded px-2 py-1" value={seed} onChange={(e)=>setSeed(parseInt(e.target.value||"0"))} />
        </HelpRow>

        <div className="flex items-center gap-2">
          <button className="px-3 py-2 rounded bg-indigo-600 hover:bg-indigo-500" onClick={()=>setAutoRotate((v)=>!v)}>{autoRotate?"Pause":"Resume"}</button>
          <button className="px-3 py-2 rounded bg-slate-700 hover:bg-slate-600" onClick={()=>fgRef.current?.zoomToFit(800,50,()=>true)}>Recenter</button>
        </div>

        <div className={`mt-4 grid grid-cols-2 gap-3 text-sm rounded ${isPercolating ? "animate-[pulseFlash_1.8s_ease-in-out_infinite] border border-emerald-400/60" : ""}`}>
          <Stat label="Edges" value={edgeCount} />
          <Stat label="Density" value={density.toFixed(3)} />
          <Stat label="Giant component" value={`${giant} (${Math.round(giantRatio*100)}%)`} />
          <Stat label="Words" value={words.length} />
        </div>

        <div className="mt-4 text-xs text-slate-300 leading-relaxed">
          <p className="mb-2"><b>Tip:</b> To spot the percolation point, increase <i>Vocabulary size</i> slowly, or lower the <i>Similarity threshold</i>. When the largest cluster suddenly covers most words (≈60%+), the panel above will pulse to signal the transition.</p>
        </div>
      </div>

      {/* 3D Graph */}
      <div className="col-span-9 relative">
        <ForceGraph3D
          ref={fgRef}
          graphData={graph}
          backgroundColor="#0f172a"
          nodeLabel={(n) => n.id}
          nodeThreeObject={(n) => {
            const group = new THREE.Group();
            const sphere = new THREE.Mesh(
              new THREE.SphereGeometry(3.2, 16, 16),
              new THREE.MeshStandardMaterial({ color: colorByIndex(n.idx) })
            );
            group.add(sphere);

            // text sprite above the node for readability
            const canvas = document.createElement("canvas");
            const size = 256;
            canvas.width = size; canvas.height = size;
            const ctx = canvas.getContext("2d");
            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 56px Inter, system-ui, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(n.id, size/2, size/2);
            const tex = new THREE.CanvasTexture(canvas);
            tex.minFilter = THREE.LinearFilter;
            const spriteMat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
            const sprite = new THREE.Sprite(spriteMat);
            sprite.position.set(0, 6.5, 0);
            const scale = 12 + Math.min(n.id.length, 8);
            sprite.scale.set(scale * 3, scale, 1);
            group.add(sprite);
            return group;
          }}
          linkColor={() => "#94a3b8"}
          linkOpacity={0.55}
          linkWidth={(l) => Math.max(0.2, (l.value - threshold) * 3)}
          onEngineTick={() => {}}
          onEngineStop={() => {
            try { fgRef.current?.zoomToFit(600, 40, () => true); } catch {}
          }}
          enableNodeDrag={false}
          onNodeClick={(node) => {
            const distance = 60;
            const controls = fgRef.current?.controls?.();
            const camera = controls?.object;
            if (!camera) return;
            const distRatio = 1 + distance / Math.hypot(node.x || 1, node.y || 1, node.z || 1);
            camera.position.set((node.x || 0) * distRatio, (node.y || 0) * distRatio, (node.z || 0) * distRatio);
            controls?.target.set(node.x || 0, node.y || 0, node.z || 0);
          }}
          showNavInfo={false}
          onRendererInitialized={() => {
            const scene = fgRef.current?.scene?.();
            if (!scene) return;
            scene.add(new THREE.AmbientLight(0xffffff, 0.55));
            const dir = new THREE.DirectionalLight(0xffffff, 0.7);
            dir.position.set(80, 120, 100);
            scene.add(dir);
          }}
          d3Force={(fg) => forceInit(fg)}
        />

        {/* Legend/notice */}
        <div className="absolute bottom-3 left-3 text-xs text-slate-300 bg-slate-800/70 backdrop-blur px-3 py-2 rounded">
          Unique color per word • Click a node to center • Use mouse to orbit/zoom
        </div>

        {isPercolating && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded bg-emerald-600 text-xs">
            Percolation threshold crossed
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="bg-slate-800/60 rounded p-3">
      <div className="text-slate-400 text-[11px] uppercase tracking-wide">{label}</div>
      <div className="text-sm mt-1">{value}</div>
    </div>
  );
}

function HelpRow({ label, desc, children }) {
  return (
    <div className="bg-slate-800/60 rounded p-3">
      <div className="flex items-center gap-2 mb-2">
        <div className="font-medium">{label}</div>
        <HelpIcon desc={desc} />
      </div>
      {children}
    </div>
  );
}

function HelpIcon({ desc }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button
        className="w-5 h-5 rounded-full bg-slate-700 hover:bg-slate-600 text-[11px] flex items-center justify-center"
        onClick={() => setOpen((v) => !v)}
        aria-label="Help"
      >
        ?
      </button>
      {open && (
        <div className="absolute z-20 mt-2 w-64 p-3 bg-slate-900 border border-slate-700 rounded shadow-xl text-xs">
          {desc}
          <div className="text-right mt-2">
            <button className="text-indigo-300 hover:text-indigo-200" onClick={() => setOpen(false)}>Got it</button>
          </div>
        </div>
      )}
    </div>
  );
}
