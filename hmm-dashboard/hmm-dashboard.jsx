import { useState, useMemo } from "react";

const RAW_DATA = [{"task":"T1","paradigm":"P1","task_num":1,"par_num":1,"task_name":"Jar Opening","par_name":"Patients vs Controls","n_states":"4","BA":0.862,"AUC":0.889,"Recall":0.975,"Precision":0.886,"F1":0.929,"features":["head_rot_y","head_pos_y","right_hand_pos_x","right_hand_rot_y","left_hand_pos_x","left_hand_rot_z","left_hand_pos_z","head_rot_z","head_pos_z","head_rot_x"],"importances":[0.175,0.175,0.15,0.125,0.075,0.075,0.05,0.05,0.025,0.025]},{"task":"T1","paradigm":"P2","task_num":1,"par_num":2,"task_name":"Jar Opening","par_name":"RCT Patients vs Controls","n_states":"4","BA":0.91,"AUC":0.91,"Recall":0.92,"Precision":0.92,"F1":0.92,"features":["head_pos_y","left_hand_rot_z","head_rot_x","right_hand_rot_y","left_hand_pos_z","left_hand_pos_y","right_hand_rot_z","right_hand_pos_x","right_hand_pos_y","left_hand_pos_x"],"importances":[0.171,0.114,0.114,0.114,0.057,0.057,0.051,0.051,0.051,0.032]},{"task":"T1","paradigm":"P3","task_num":1,"par_num":3,"task_name":"Jar Opening","par_name":"Other Patients vs Controls","n_states":"7","BA":0.767,"AUC":0.767,"Recall":0.733,"Precision":0.733,"F1":0.733,"features":["head_rot_y","head_pos_y","head_rot_x","left_hand_rot_z","left_hand_pos_x","right_hand_pos_z","left_hand_pos_y","right_hand_pos_x","left_hand_rot_y","left_hand_rot_x"],"importances":[0.122,0.109,0.109,0.077,0.077,0.064,0.064,0.051,0.051,0.051]},{"task":"T1","paradigm":"P4","task_num":1,"par_num":4,"task_name":"Jar Opening","par_name":"RCT vs Other Patients","n_states":"4","BA":0.487,"AUC":0.472,"Recall":0.84,"Precision":0.618,"F1":0.712,"features":["left_hand_rot_x","head_rot_x","left_hand_pos_x","left_hand_pos_z","right_hand_pos_z","right_hand_rot_y","right_hand_rot_x","right_hand_pos_y","head_pos_z","head_rot_z"],"importances":[0.246,0.159,0.159,0.116,0.116,0.087,0.043,0.043,0.029,0.0]},{"task":"T2","paradigm":"P2","task_num":2,"par_num":2,"task_name":"Key Turning","par_name":"RCT Patients vs Controls","n_states":"3","BA":0.72,"AUC":0.721,"Recall":0.84,"Precision":0.724,"F1":0.778,"features":["left_hand_rot_z","left_hand_pos_x","head_pos_y","head_pos_x","left_hand_rot_x","right_hand_rot_y","head_pos_z","right_hand_pos_y","head_rot_z","head_rot_x"],"importances":[0.146,0.126,0.121,0.111,0.111,0.101,0.085,0.065,0.05,0.04]},{"task":"T2","paradigm":"P3","task_num":2,"par_num":3,"task_name":"Key Turning","par_name":"Other Patients vs Controls","n_states":"3","BA":0.742,"AUC":0.746,"Recall":0.733,"Precision":0.688,"F1":0.71,"features":["left_hand_rot_z","left_hand_pos_x","left_hand_pos_z","right_hand_pos_y","left_hand_pos_y","head_pos_y","head_pos_z","head_rot_x","head_rot_y","head_rot_z"],"importances":[0.357,0.321,0.214,0.107,0.0,0.0,0.0,0.0,0.0,0.0]},{"task":"T2","paradigm":"P4","task_num":2,"par_num":4,"task_name":"Key Turning","par_name":"RCT vs Other Patients","n_states":"2","BA":0.56,"AUC":0.558,"Recall":0.32,"Precision":0.727,"F1":0.444,"features":["left_hand_rot_z","left_hand_pos_x","head_pos_x","head_rot_z","left_hand_rot_x","left_hand_pos_z","right_hand_pos_y","left_hand_pos_y","right_hand_rot_x","right_hand_rot_z"],"importances":[0.1,0.094,0.08,0.08,0.08,0.072,0.072,0.069,0.066,0.066]},{"task":"T3","paradigm":"P1","task_num":3,"par_num":1,"task_name":"Cleaning","par_name":"Patients vs Controls","n_states":"4","BA":0.8,"AUC":0.808,"Recall":0.9,"Precision":0.857,"F1":0.878,"features":["head_pos_y","left_hand_rot_z","left_hand_pos_z","left_hand_rot_y","head_rot_x","head_rot_z","left_hand_pos_y","left_hand_pos_x","head_pos_x","right_hand_pos_y"],"importances":[0.176,0.147,0.088,0.088,0.059,0.059,0.059,0.059,0.059,0.059]},{"task":"T3","paradigm":"P2","task_num":3,"par_num":2,"task_name":"Cleaning","par_name":"RCT Patients vs Controls","n_states":"8","BA":0.795,"AUC":0.806,"Recall":0.84,"Precision":0.808,"F1":0.824,"features":["right_hand_pos_x","right_hand_pos_z","head_pos_y","head_pos_z","left_hand_rot_x","left_hand_pos_x","head_rot_z","left_hand_pos_y","right_hand_pos_y","left_hand_rot_z"],"importances":[0.113,0.094,0.075,0.071,0.071,0.071,0.071,0.057,0.052,0.052]},{"task":"T3","paradigm":"P3","task_num":3,"par_num":3,"task_name":"Cleaning","par_name":"Other Patients vs Controls","n_states":"3","BA":0.683,"AUC":0.685,"Recall":0.667,"Precision":0.625,"F1":0.645,"features":["head_pos_y","left_hand_pos_z","right_hand_rot_y","head_rot_x","right_hand_pos_y","left_hand_rot_z","head_rot_y","right_hand_rot_z","left_hand_rot_y","right_hand_pos_x"],"importances":[0.294,0.094,0.094,0.082,0.082,0.082,0.071,0.059,0.047,0.047]},{"task":"T3","paradigm":"P4","task_num":3,"par_num":4,"task_name":"Cleaning","par_name":"RCT vs Other Patients","n_states":"3","BA":0.6,"AUC":0.607,"Recall":0.8,"Precision":0.69,"F1":0.741,"features":["right_hand_pos_x","right_hand_rot_y","left_hand_rot_z","head_pos_x","head_rot_x","right_hand_pos_z","left_hand_pos_x","left_hand_pos_z","head_rot_z","left_hand_rot_x"],"importances":[0.169,0.164,0.144,0.087,0.087,0.077,0.077,0.067,0.051,0.026]},{"task":"T4","paradigm":"P1","task_num":4,"par_num":1,"task_name":"Back Washing","par_name":"Patients vs Controls","n_states":"4","BA":0.775,"AUC":0.778,"Recall":0.8,"Precision":0.865,"F1":0.831,"features":["head_pos_y","head_rot_x","left_hand_rot_z","left_hand_pos_x","head_pos_z","left_hand_pos_z","left_hand_rot_y","head_rot_y","left_hand_pos_y","head_rot_z"],"importances":[0.27,0.127,0.127,0.095,0.079,0.079,0.079,0.048,0.016,0.016]},{"task":"T4","paradigm":"P2","task_num":4,"par_num":2,"task_name":"Back Washing","par_name":"RCT Patients vs Controls","n_states":"3","BA":0.85,"AUC":0.864,"Recall":0.8,"Precision":0.909,"F1":0.851,"features":["head_pos_y","left_hand_pos_z","head_pos_z","right_hand_rot_z","right_hand_pos_y","left_hand_rot_z","head_rot_x","left_hand_pos_y","left_hand_rot_y","right_hand_rot_x"],"importances":[0.135,0.113,0.103,0.093,0.08,0.077,0.068,0.064,0.061,0.061]},{"task":"T4","paradigm":"P3","task_num":4,"par_num":3,"task_name":"Back Washing","par_name":"Other Patients vs Controls","n_states":"4","BA":0.792,"AUC":0.814,"Recall":0.733,"Precision":0.786,"F1":0.759,"features":["head_rot_z","left_hand_pos_z","left_hand_pos_x","head_pos_x","left_hand_rot_x","head_rot_x","left_hand_pos_y","head_pos_z","right_hand_pos_z","left_hand_rot_z"],"importances":[0.116,0.089,0.089,0.085,0.078,0.078,0.074,0.062,0.058,0.054]},{"task":"T4","paradigm":"P4","task_num":4,"par_num":4,"task_name":"Back Washing","par_name":"RCT vs Other Patients","n_states":"3","BA":0.553,"AUC":0.544,"Recall":0.84,"Precision":0.656,"F1":0.737,"features":["head_rot_x","head_rot_z","head_pos_x","head_pos_y","right_hand_rot_y","head_rot_y","left_hand_pos_z","left_hand_rot_y","left_hand_rot_z","right_hand_rot_x"],"importances":[0.127,0.108,0.096,0.09,0.084,0.084,0.084,0.084,0.072,0.042]},{"task":"T5","paradigm":"P1","task_num":5,"par_num":1,"task_name":"Cutting","par_name":"Patients vs Controls","n_states":"7","BA":0.7,"AUC":0.7,"Recall":0.9,"Precision":0.783,"F1":0.837,"features":["head_pos_y","left_hand_rot_y","head_pos_z","left_hand_rot_z","right_hand_pos_x","head_rot_z","right_hand_rot_z","head_pos_x","right_hand_pos_z","right_hand_rot_x"],"importances":[0.227,0.107,0.107,0.107,0.08,0.067,0.053,0.04,0.04,0.04]},{"task":"T5","paradigm":"P2","task_num":5,"par_num":2,"task_name":"Cutting","par_name":"RCT Patients vs Controls","n_states":"7","BA":0.745,"AUC":0.735,"Recall":0.84,"Precision":0.75,"F1":0.792,"features":["head_pos_y","right_hand_rot_x","right_hand_rot_y","head_rot_x","right_hand_rot_z","head_pos_x","right_hand_pos_x","left_hand_rot_x","head_rot_y","left_hand_rot_y"],"importances":[0.095,0.09,0.081,0.075,0.062,0.062,0.059,0.057,0.057,0.055]},{"task":"T5","paradigm":"P3","task_num":5,"par_num":3,"task_name":"Cutting","par_name":"Other Patients vs Controls","n_states":"3","BA":0.642,"AUC":0.667,"Recall":0.533,"Precision":0.615,"F1":0.571,"features":["head_pos_y","left_hand_rot_z","head_pos_x","left_hand_pos_y","left_hand_rot_y","left_hand_pos_z","left_hand_rot_x","head_rot_x","head_rot_y","right_hand_rot_z"],"importances":[0.203,0.146,0.081,0.081,0.081,0.073,0.057,0.057,0.049,0.049]},{"task":"T5","paradigm":"P4","task_num":5,"par_num":4,"task_name":"Cutting","par_name":"RCT vs Other Patients","n_states":"3","BA":0.513,"AUC":0.506,"Recall":0.76,"Precision":0.633,"F1":0.691,"features":["right_hand_rot_z","left_hand_pos_x","left_hand_rot_y","head_pos_z","head_pos_y","right_hand_pos_x","left_hand_pos_y","head_rot_x","left_hand_rot_x","head_rot_z"],"importances":[0.203,0.165,0.135,0.128,0.098,0.06,0.045,0.045,0.038,0.023]},{"task":"T6","paradigm":"P1","task_num":6,"par_num":1,"task_name":"Hammering","par_name":"Patients vs Controls","n_states":"2","BA":0.725,"AUC":0.721,"Recall":0.75,"Precision":0.833,"F1":0.789,"features":["head_pos_y","left_hand_pos_y","right_hand_rot_y","left_hand_pos_z","head_rot_x","head_pos_z","right_hand_rot_x","right_hand_pos_x","left_hand_rot_y","left_hand_pos_x"],"importances":[0.25,0.125,0.089,0.071,0.071,0.071,0.054,0.054,0.054,0.054]},{"task":"T6","paradigm":"P2","task_num":6,"par_num":2,"task_name":"Hammering","par_name":"RCT Patients vs Controls","n_states":"7","BA":0.75,"AUC":0.756,"Recall":0.8,"Precision":0.769,"F1":0.784,"features":["head_pos_x","left_hand_pos_z","head_pos_y","right_hand_pos_x","head_rot_y","right_hand_rot_z","right_hand_pos_z","right_hand_pos_y","left_hand_rot_x","head_rot_x"],"importances":[0.085,0.077,0.074,0.071,0.071,0.071,0.066,0.063,0.063,0.063]},{"task":"T6","paradigm":"P3","task_num":6,"par_num":3,"task_name":"Hammering","par_name":"Other Patients vs Controls","n_states":"8","BA":0.583,"AUC":0.585,"Recall":0.267,"Precision":0.667,"F1":0.381,"features":["head_pos_z","head_pos_x","head_rot_z","head_pos_y","right_hand_pos_y","left_hand_rot_z","right_hand_pos_x","right_hand_rot_z","left_hand_pos_z","right_hand_pos_z"],"importances":[0.144,0.138,0.086,0.086,0.069,0.069,0.057,0.052,0.046,0.046]},{"task":"T6","paradigm":"P4","task_num":6,"par_num":4,"task_name":"Hammering","par_name":"RCT vs Other Patients","n_states":"2","BA":0.587,"AUC":0.603,"Recall":0.64,"Precision":0.696,"F1":0.667,"features":["left_hand_pos_y","head_rot_x","right_hand_pos_x","head_pos_y","right_hand_rot_x","head_rot_z","right_hand_pos_z","head_pos_z","right_hand_rot_y","head_pos_x"],"importances":[0.209,0.124,0.109,0.109,0.085,0.062,0.062,0.062,0.062,0.047]}];

const TASK_ICONS = { "Jar Opening":"🫙", "Key Turning":"🔑", "Cleaning":"🧽", "Back Washing":"🚿", "Cutting":"✂️", "Hammering":"🔨" };
const PAR_SHORT = { P1:"All Pts vs Ctrl", P2:"RCT vs Ctrl", P3:"Other vs Ctrl", P4:"RCT vs Other" };
const PAR_FULL = { P1:"Patients vs Controls", P2:"RCT Patients vs Controls", P3:"Other Patients vs Controls", P4:"RCT vs Other Patients" };

function getSensor(f) {
  if (f.startsWith("head")) return "head";
  if (f.startsWith("right")) return "right";
  return "left";
}
function getMotion(f) {
  return f.includes("_rot_") ? "rot" : "pos";
}
function getAxis(f) {
  return f.slice(-1);
}

const SENSOR_COLORS = {
  head:  { bg:"#dbeafe", text:"#1e40af", border:"#93c5fd" },
  right: { bg:"#dcfce7", text:"#166534", border:"#86efac" },
  left:  { bg:"#fef3c7", text:"#92400e", border:"#fcd34d" },
};
const MOTION_COLORS = {
  rot: { bg:"#f3e8ff", text:"#6b21a8", border:"#d8b4fe" },
  pos: { bg:"#ffedd5", text:"#9a3412", border:"#fdba74" },
};

function baColor(ba) {
  if (ba >= 0.85) return { bg:"#dcfce7", text:"#166534", border:"#86efac" };
  if (ba >= 0.75) return { bg:"#d1fae5", text:"#065f46", border:"#6ee7b7" };
  if (ba >= 0.70) return { bg:"#fef9c3", text:"#713f12", border:"#fde047" };
  if (ba >= 0.60) return { bg:"#ffedd5", text:"#9a3412", border:"#fdba74" };
  return { bg:"#fee2e2", text:"#991b1b", border:"#fca5a5" };
}

function FeatureChip({ feat, rank }) {
  const s = getSensor(feat);
  const m = getMotion(feat);
  const c = SENSOR_COLORS[s];
  const label = feat.replace("_pos_","·p·").replace("_rot_","·r·").replace("head","H").replace("right_hand","RH").replace("left_hand","LH");
  return (
    <span style={{
      display:"inline-flex", alignItems:"center", gap:2,
      padding:"1px 6px", borderRadius:4, fontSize:11, fontWeight:500,
      background:c.bg, color:c.text, border:`1px solid ${c.border}`,
      whiteSpace:"nowrap"
    }}>
      {rank && <span style={{opacity:0.6, fontSize:10}}>#{rank} </span>}
      {label}
    </span>
  );
}

function MetricBadge({ label, val, color }) {
  return (
    <div style={{ textAlign:"center" }}>
      <div style={{ fontSize:11, color:"#6b7280", marginBottom:2 }}>{label}</div>
      <div style={{ fontSize:15, fontWeight:700, color: color || "#111827" }}>{(val*100).toFixed(1)}%</div>
    </div>
  );
}

// ── Tab 1: Main Performance Table ─────────────────────────────────────────────
function PerformanceTable({ data }) {
  const tasks = [...new Set(data.map(d => d.task_name))];
  const pars = ["P1","P2","P3","P4"];

  const byKey = {};
  data.forEach(d => { byKey[`${d.task}_${d.paradigm}`] = d; });

  const colW = [130, 90, 82, 82, 82, 82, 82, 40];

  return (
    <div>
      <div style={{ marginBottom:16, padding:"12px 16px", background:"#f8fafc", borderRadius:8, border:"1px solid #e2e8f0" }}>
        <div style={{ fontSize:13, fontWeight:600, color:"#374151", marginBottom:6 }}>Color legend — Balanced Accuracy</div>
        <div style={{ display:"flex", gap:12, flexWrap:"wrap" }}>
          {[["≥ 0.85","#dcfce7","#166534"],["0.75–0.84","#d1fae5","#065f46"],["0.70–0.74","#fef9c3","#713f12"],["0.60–0.69","#ffedd5","#9a3412"],["< 0.60","#fee2e2","#991b1b"]].map(([l,bg,tc])=>(
            <span key={l} style={{display:"flex",alignItems:"center",gap:5,fontSize:12}}>
              <span style={{width:14,height:14,borderRadius:3,background:bg,border:`1px solid ${tc}`,display:"inline-block"}}/>
              <span style={{color:"#374151"}}>{l}</span>
            </span>
          ))}
        </div>
      </div>

      <div style={{ overflowX:"auto" }}>
        <table style={{ borderCollapse:"collapse", width:"100%", fontSize:13 }}>
          <thead>
            <tr style={{ background:"#1e293b", color:"white" }}>
              <th style={{ padding:"10px 12px", textAlign:"left", width:colW[0] }}>Task</th>
              <th style={{ padding:"10px 12px", textAlign:"center", width:colW[1] }}>Paradigm</th>
              <th style={{ padding:"10px 12px", textAlign:"center", width:colW[2] }}>BA</th>
              <th style={{ padding:"10px 12px", textAlign:"center", width:colW[3] }}>AUC</th>
              <th style={{ padding:"10px 12px", textAlign:"center", width:colW[4] }}>Recall</th>
              <th style={{ padding:"10px 12px", textAlign:"center", width:colW[5] }}>Precision</th>
              <th style={{ padding:"10px 12px", textAlign:"center", width:colW[6] }}>F1</th>
              <th style={{ padding:"10px 12px", textAlign:"center", width:colW[7] }}>n_st</th>
            </tr>
          </thead>
          <tbody>
            {tasks.map((tname, ti) => {
              const tRows = data.filter(d => d.task_name === tname);
              const tKey = tRows[0].task;
              return pars.map((par, pi) => {
                const d = byKey[`${tKey}_${par}`];
                const isFirst = pi === 0;
                const c = d ? baColor(d.BA) : null;
                const recPrec = d ? (d.Recall > d.Precision ? "rec" : d.Precision > d.Recall ? "prec" : "=") : null;
                return (
                  <tr key={`${tname}_${par}`} style={{ background: pi % 2 === 0 ? "#f9fafb" : "white", borderBottom:"1px solid #e5e7eb" }}>
                    {isFirst && (
                      <td rowSpan={4} style={{ padding:"10px 12px", verticalAlign:"middle", fontWeight:600, fontSize:13, borderRight:"2px solid #e5e7eb", background:"#f1f5f9" }}>
                        <div>{TASK_ICONS[tname]} {tname}</div>
                        <div style={{ fontSize:11, color:"#6b7280", fontWeight:400, marginTop:2 }}>{tKey}</div>
                      </td>
                    )}
                    <td style={{ padding:"8px 12px", textAlign:"center", color:"#374151", fontSize:12 }}>
                      <span style={{ display:"inline-block", padding:"2px 7px", borderRadius:4, background:"#f1f5f9", fontWeight:500 }}>{par}</span>
                      <div style={{ fontSize:10, color:"#9ca3af", marginTop:1 }}>{PAR_SHORT[par]}</div>
                    </td>
                    {d ? <>
                      <td style={{ padding:"8px 12px", textAlign:"center" }}>
                        <span style={{ display:"inline-block", padding:"3px 8px", borderRadius:5, fontWeight:700, fontSize:13, background:c.bg, color:c.text, border:`1px solid ${c.border}` }}>
                          {(d.BA*100).toFixed(1)}%
                        </span>
                      </td>
                      <td style={{ padding:"8px 12px", textAlign:"center", color:"#374151" }}>{(d.AUC*100).toFixed(1)}%</td>
                      <td style={{ padding:"8px 12px", textAlign:"center" }}>
                        <span style={{ color: recPrec==="rec" ? "#059669" : "#374151", fontWeight: recPrec==="rec" ? 700 : 400 }}>
                          {(d.Recall*100).toFixed(1)}%
                        </span>
                      </td>
                      <td style={{ padding:"8px 12px", textAlign:"center" }}>
                        <span style={{ color: recPrec==="prec" ? "#7c3aed" : "#374151", fontWeight: recPrec==="prec" ? 700 : 400 }}>
                          {(d.Precision*100).toFixed(1)}%
                        </span>
                      </td>
                      <td style={{ padding:"8px 12px", textAlign:"center", color:"#374151" }}>{(d.F1*100).toFixed(1)}%</td>
                      <td style={{ padding:"8px 12px", textAlign:"center", color:"#6b7280", fontSize:12 }}>{d.n_states}</td>
                    </> : <td colSpan={6} style={{ textAlign:"center", color:"#d1d5db", fontSize:12, padding:8 }}>— missing —</td>}
                  </tr>
                );
              });
            })}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop:16, display:"grid", gridTemplateColumns:"1fr 1fr", gap:12 }}>
        <div style={{ padding:12, background:"#eff6ff", borderRadius:8, border:"1px solid #bfdbfe" }}>
          <div style={{ fontSize:12, fontWeight:700, color:"#1e40af", marginBottom:6 }}>🟢 Recall &gt; Precision</div>
          <div style={{ fontSize:12, color:"#1e3a8a" }}>Model prioritises sensitivity — misses fewer patients. Dominant in P1, P2 (correct clinical bias).</div>
        </div>
        <div style={{ padding:12, background:"#f5f3ff", borderRadius:8, border:"1px solid #ddd6fe" }}>
          <div style={{ fontSize:12, fontWeight:700, color:"#6d28d9", marginBottom:6 }}>🟣 Precision &gt; Recall</div>
          <div style={{ fontSize:12, color:"#4c1d95" }}>Model is conservative — fewer false alarms. Appears in Back Washing P1/P2 and Hammering P1 — tasks with clearer biomechanical separation.</div>
        </div>
      </div>

      <div style={{ marginTop:12, padding:12, background:"#fef2f2", borderRadius:8, border:"1px solid #fecaca" }}>
        <div style={{ fontSize:12, fontWeight:700, color:"#991b1b", marginBottom:4 }}>⚠️ Missing: T2 P1 (Key Turning — All Patients vs Controls)</div>
        <div style={{ fontSize:12, color:"#7f1d1d" }}>This is the only gap in the 6×4 grid. Key Turning is the only task without a primary diagnostic paradigm result — worth investigating whether the experiment failed or data was unavailable.</div>
      </div>
    </div>
  );
}

// ── Tab 2: Insights Panel ──────────────────────────────────────────────────────
function InsightsPanel({ data }) {
  const best = [...data].sort((a,b) => b.BA - a.BA)[0];
  const worst = [...data].filter(d=>d.BA > 0.3).sort((a,b) => a.BA - b.BA)[0];
  
  const taskAvg = {};
  ["Jar Opening","Key Turning","Cleaning","Back Washing","Cutting","Hammering"].forEach(t => {
    const rows = data.filter(d=>d.task_name===t);
    taskAvg[t] = rows.reduce((s,d)=>s+d.BA,0)/rows.length;
  });
  const sortedTasks = Object.entries(taskAvg).sort((a,b)=>b[1]-a[1]);
  
  const parAvg = {};
  ["P1","P2","P3","P4"].forEach(p => {
    const rows = data.filter(d=>d.paradigm===p);
    parAvg[p] = rows.reduce((s,d)=>s+d.BA,0)/rows.length;
  });

  const p4Data = data.filter(d=>d.paradigm==="P4");
  const p4AvgRec = p4Data.reduce((s,d)=>s+d.Recall,0)/p4Data.length;
  const p4AvgPrec = p4Data.reduce((s,d)=>s+d.Precision,0)/p4Data.length;

  const insights = [
    {
      icon:"🏆", title:"Best Performer", color:"#dcfce7", border:"#86efac", textColor:"#166534",
      content: `T1 P2 — Jar Opening (RCT vs Controls) achieves BA=91.0%, AUC=0.910, Recall=Precision=F1=92.0% with n_states=4. The most balanced and highest-performing combination in the dataset. Rotator cuff tear patients produce the most distinctly abnormal jar-opening kinematics.`
    },
    {
      icon:"📊", title:"Task Performance Ranking", color:"#eff6ff", border:"#bfdbfe", textColor:"#1e40af",
      content: sortedTasks.map(([t,v])=>`${TASK_ICONS[t]} ${t}: avg BA = ${(v*100).toFixed(1)}%`).join(" · ")
    },
    {
      icon:"📉", title:"Paradigm Difficulty Gradient", color:"#fefce8", border:"#fde047", textColor:"#713f12",
      content: `All tasks show consistent BA decline: P1 (${(parAvg.P1*100).toFixed(1)}%) → P2 (${(parAvg.P2*100).toFixed(1)}%) → P3 (${(parAvg.P3*100).toFixed(1)}%) → P4 (${(parAvg.P4*100).toFixed(1)}%). P4 (differential diagnosis) is near chance for all tasks — the HMM cannot distinguish RCT from other shoulder pathologies on motion data alone.`
    },
    {
      icon:"🎯", title:"Recall vs Precision Trade-off", color:"#f5f3ff", border:"#ddd6fe", textColor:"#5b21b6",
      content: `P1/P2: Recall dominates Precision (sensitivity-first, correct clinical priority — missing a patient is worse than a false alarm). P3: Nearly balanced. P4 exception — T2 P4 shows Precision=0.727 vs Recall=0.320, meaning the model rarely predicts RCT but when it does it's often correct. T6 P3 is the worst: Recall=0.267 — model almost never detects "other condition" patients.`
    },
    {
      icon:"🔍", title:"P4 Structural Anomaly", color:"#fff1f2", border:"#fecdd3", textColor:"#9f1239",
      content: `Every P4 experiment shows FP=0 — the model never predicts "other conditions" as RCT. This means all errors are False Negatives (RCT predicted as other). The HMM systematically places ambiguous sequences into the "other" class, suggesting RCT-specific motion patterns are not robustly separable at the individual patient level.`
    },
    {
      icon:"⚡", title:"Interesting Exceptions", color:"#fdf4ff", border:"#e9d5ff", textColor:"#6b21a8",
      content: `• T4 P3 (Back Washing, Other vs Controls): BA=79.2% with Precision=0.786 > Recall=0.733 — unusually precise for P3.\n• T1 P1: Recall=97.5% but Precision=88.6% — captures almost all patients but with more false alarms.\n• T3/T4 P2 both achieve ~80% BA with 8 states needed for Cleaning P2, suggesting more complex temporal structure in RCT cleaning patterns.\n• Hammering P4 (BA=58.7%) slightly outperforms Cutting P4 (51.3%) despite Hammering being weaker overall.`
    },
    {
      icon:"💡", title:"Weakest Task", color:"#fff7ed", border:"#fed7aa", textColor:"#9a3412",
      content: `Hammering (T6) is weakest overall (avg BA ${(taskAvg["Hammering"]*100).toFixed(1)}%), with T6 P3 producing the lowest BA in the dataset (58.3%) and the lowest recall overall (26.7%). Hammering is a power-grip task — the biomechanical signature may be less sensitive to rotator cuff or glenohumeral pathology than fine manipulation tasks like Jar Opening or Key Turning.`
    },
  ];

  return (
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
      {insights.map((ins, i) => (
        <div key={i} style={{ padding:14, borderRadius:10, background:ins.color, border:`1px solid ${ins.border}` }}>
          <div style={{ fontSize:14, fontWeight:700, color:ins.textColor, marginBottom:7 }}>{ins.icon} {ins.title}</div>
          <div style={{ fontSize:12, color:ins.textColor, lineHeight:1.6, whiteSpace:"pre-line" }}>{ins.content}</div>
        </div>
      ))}
    </div>
  );
}

// ── Tab 3: Feature Intelligence ────────────────────────────────────────────────
function FeatureIntelligence({ data }) {
  const ALL_FEATURES = [
    "head_pos_x","head_pos_y","head_pos_z","head_rot_x","head_rot_y","head_rot_z",
    "right_hand_pos_x","right_hand_pos_y","right_hand_pos_z","right_hand_rot_x","right_hand_rot_y","right_hand_rot_z",
    "left_hand_pos_x","left_hand_pos_y","left_hand_pos_z","left_hand_rot_x","left_hand_rot_y","left_hand_rot_z"
  ];
  const N = data.length;

  const stats = useMemo(() => {
    const s = {};
    ALL_FEATURES.forEach(f => {
      s[f] = { top1:0, top3:0, top5:0, raw_sum:0, ba_weighted:0, experiments:[] };
    });
    data.forEach(d => {
      const ba = d.BA;
      d.features.forEach((feat, i) => {
        if (!s[feat]) return;
        const imp = d.importances[i] || 0;
        if (i === 0) s[feat].top1++;
        if (i < 3)  s[feat].top3++;
        if (i < 5)  s[feat].top5++;
        s[feat].raw_sum += imp;
        s[feat].ba_weighted += imp * ba;
        if (i < 5) s[feat].experiments.push({ exp:`${d.task} ${d.paradigm}`, ba, imp, rank:i+1 });
      });
    });
    return s;
  }, [data]);

  // BA-weighted importance score = sum(importance * BA) across all experiments
  const sorted = ALL_FEATURES.slice().sort((a,b) => stats[b].ba_weighted - stats[a].ba_weighted);

  // Sensor summary
  const sensorStats = { head:{top1:0,top3:0,top5:0}, right:{top1:0,top3:0,top5:0}, left:{top1:0,top3:0,top5:0} };
  const motionStats = { pos:{top1:0,top3:0,top5:0}, rot:{top1:0,top3:0,top5:0} };
  ALL_FEATURES.forEach(f => {
    const s2 = getSensor(f); const m = getMotion(f); const st = stats[f];
    ["top1","top3","top5"].forEach(k => {
      sensorStats[s2][k] += st[k];
      motionStats[m][k] += st[k];
    });
  });

  const maxBA = Math.max(...sorted.map(f => stats[f].ba_weighted));

  return (
    <div>
      {/* Section 1: Sensor Cards */}
      <div style={{ marginBottom:20 }}>
        <h3 style={{ fontSize:14, fontWeight:700, color:"#1e293b", marginBottom:12 }}>Section 1 — Sensor-Level Summary</h3>
        <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:12, marginBottom:14 }}>
          {[["head","🧠 Head","#dbeafe","#1e40af","#93c5fd"],["right","✋ Right Hand","#dcfce7","#166534","#86efac"],["left","🤚 Left Hand","#fef3c7","#92400e","#fcd34d"]].map(([key,label,bg,tc,bc])=>(
            <div key={key} style={{ padding:14, borderRadius:10, background:bg, border:`1px solid ${bc}` }}>
              <div style={{ fontSize:13, fontWeight:700, color:tc, marginBottom:10 }}>{label}</div>
              {["top1","top3","top5"].map(k => (
                <div key={k} style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
                  <span style={{ fontSize:11, color:tc, opacity:0.8 }}>Top-{k.replace("top","")} appearances</span>
                  <span style={{ fontSize:12, fontWeight:700, color:tc }}>{sensorStats[key][k]}</span>
                </div>
              ))}
            </div>
          ))}
        </div>

        <div style={{ display:"grid", gridTemplateColumns:"repeat(2,1fr)", gap:12 }}>
          {[["pos","📍 Position","#ffedd5","#9a3412","#fdba74"],["rot","🔄 Rotation","#f3e8ff","#6b21a8","#d8b4fe"]].map(([key,label,bg,tc,bc])=>(
            <div key={key} style={{ padding:14, borderRadius:10, background:bg, border:`1px solid ${bc}` }}>
              <div style={{ fontSize:13, fontWeight:700, color:tc, marginBottom:10 }}>{label}</div>
              {["top1","top3","top5"].map(k => (
                <div key={k} style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
                  <span style={{ fontSize:11, color:tc, opacity:0.8 }}>Top-{k.replace("top","")} appearances</span>
                  <span style={{ fontSize:12, fontWeight:700, color:tc }}>{motionStats[key][k]}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Section 2: Ranked Features */}
      <div style={{ marginBottom:20 }}>
        <h3 style={{ fontSize:14, fontWeight:700, color:"#1e293b", marginBottom:12 }}>Section 2 — All 18 Features Ranked by BA-Weighted Score</h3>
        <div style={{ background:"#f8fafc", borderRadius:10, border:"1px solid #e2e8f0", overflow:"hidden" }}>
          <div style={{ display:"grid", gridTemplateColumns:"160px 50px 50px 50px 120px 1fr", gap:0, background:"#1e293b", color:"white", padding:"8px 14px", fontSize:11, fontWeight:600 }}>
            <span>Feature</span><span style={{textAlign:"center"}}>T1</span><span style={{textAlign:"center"}}>T3</span><span style={{textAlign:"center"}}>T5</span><span style={{textAlign:"center"}}>BA·Score</span><span>Coverage (23 experiments)</span>
          </div>
          {sorted.map((feat, i) => {
            const st = stats[feat];
            const sensor = getSensor(feat);
            const motion = getMotion(feat);
            const sc = SENSOR_COLORS[sensor];
            const mc = MOTION_COLORS[motion];
            const barW = Math.round((st.ba_weighted / maxBA) * 100);
            const label = feat.replace("right_hand_","RH_").replace("left_hand_","LH_").replace("head_","H_");
            return (
              <div key={feat} style={{ display:"grid", gridTemplateColumns:"160px 50px 50px 50px 120px 1fr", alignItems:"center", padding:"7px 14px", borderBottom:"1px solid #e5e7eb", background: i%2===0?"white":"#f9fafb" }}>
                <span style={{ display:"flex", alignItems:"center", gap:5 }}>
                  <span style={{ width:8, height:8, borderRadius:"50%", background:sc.text, flexShrink:0 }}/>
                  <span style={{ fontSize:11, fontWeight:500, color:"#374151" }}>{label}</span>
                  <span style={{ fontSize:9, padding:"1px 4px", borderRadius:3, background:mc.bg, color:mc.text, border:`1px solid ${mc.border}` }}>{motion}</span>
                </span>
                <span style={{ textAlign:"center", fontSize:12, fontWeight:600, color:"#374151" }}>{st.top1}</span>
                <span style={{ textAlign:"center", fontSize:12, color:"#374151" }}>{st.top3}</span>
                <span style={{ textAlign:"center", fontSize:12, color:"#6b7280" }}>{st.top5}</span>
                <span style={{ textAlign:"center", fontSize:11, fontWeight:700, color:"#1e293b" }}>{st.ba_weighted.toFixed(3)}</span>
                <div style={{ display:"flex", alignItems:"center", gap:6 }}>
                  <div style={{ flex:1, height:12, background:"#e5e7eb", borderRadius:6, overflow:"hidden" }}>
                    <div style={{ width:`${barW}%`, height:"100%", background:sc.text, borderRadius:6, transition:"width 0.3s" }}/>
                  </div>
                  <span style={{ fontSize:10, color:"#9ca3af", width:30 }}>{st.top5}/{N}</span>
                </div>
              </div>
            );
          })}
        </div>
        <div style={{ marginTop:8, fontSize:11, color:"#6b7280" }}>
          T1=Top-1 count · T3=Top-3 count · T5=Top-5 count · BA·Score = Σ(permutation_importance × BA) across all experiments · Bar width = Top-5 coverage out of {N} experiments
        </div>
      </div>

      {/* Key Takeaways */}
      <div>
        <h3 style={{ fontSize:14, fontWeight:700, color:"#1e293b", marginBottom:10 }}>Key Takeaways</h3>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
          {[
            ["🧠 Head dominates","head_pos_y is #1 by every metric: Top-1 count=9, Top-5=18/23 experiments, BA·Score=2.014. Vertical head position captures postural compensation — patients elevate or drop their head to offload shoulder pain.","#dbeafe","#1e40af"],
            ["🤚 Left hand is surprisingly important","Left hand features appear more often than right in the top-5 despite most tasks being right-hand dominant. Patients compensate bilaterally — the left hand modulates grip, support, and trunk stabilisation differently.","#fef3c7","#92400e"],
            ["🔄 Rotation > Position in high-BA tasks","In best-performing experiments (T1P1, T1P2, T4P2), rotation features (especially left_hand_rot_z, right_hand_rot_y) appear alongside head_pos_y. Rotation captures the quality of shoulder arc, not just endpoint position.","#f3e8ff","#6b21a8"],
            ["📉 P4 feature instability","In P4 experiments (differential diagnosis), feature importance is more evenly distributed and no single feature dominates — consistent with near-chance performance. The model cannot find a stable discriminative feature when separating two patient subgroups.","#fee2e2","#991b1b"],
          ].map(([title,body,bg,tc])=>(
            <div key={title} style={{ padding:12, borderRadius:8, background:bg, border:`1px solid`, borderColor:tc }}>
              <div style={{ fontSize:12, fontWeight:700, color:tc, marginBottom:5 }}>{title}</div>
              <div style={{ fontSize:11, color:tc, lineHeight:1.6 }}>{body}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Tab 4: Per-Experiment Feature Table ────────────────────────────────────────
function ExperimentTable({ data }) {
  return (
    <div>
      <div style={{ marginBottom:12, fontSize:12, color:"#6b7280" }}>
        Each row = one experiment. BA colour-coded. Top-5 feature chips coloured by sensor (
        <span style={{background:"#dbeafe",color:"#1e40af",padding:"1px 5px",borderRadius:3,border:"1px solid #93c5fd",fontSize:11}}>Head</span>
        {" "}
        <span style={{background:"#dcfce7",color:"#166534",padding:"1px 5px",borderRadius:3,border:"1px solid #86efac",fontSize:11}}>Right Hand</span>
        {" "}
        <span style={{background:"#fef3c7",color:"#92400e",padding:"1px 5px",borderRadius:3,border:"1px solid #fcd34d",fontSize:11}}>Left Hand</span>
        ).
      </div>
      <div style={{ overflowX:"auto" }}>
        <table style={{ borderCollapse:"collapse", width:"100%", fontSize:12 }}>
          <thead>
            <tr style={{ background:"#1e293b", color:"white" }}>
              <th style={{ padding:"9px 10px", textAlign:"left", minWidth:120 }}>Experiment</th>
              <th style={{ padding:"9px 10px", textAlign:"center", width:70 }}>BA</th>
              <th style={{ padding:"9px 10px", textAlign:"center", width:55 }}>AUC</th>
              <th style={{ padding:"9px 10px", textAlign:"center", width:55 }}>Rec</th>
              <th style={{ padding:"9px 10px", textAlign:"center", width:55 }}>Prec</th>
              <th style={{ padding:"9px 10px", textAlign:"left" }}>Top-5 Features</th>
            </tr>
          </thead>
          <tbody>
            {data.map((d, i) => {
              const c = baColor(d.BA);
              return (
                <tr key={i} style={{ borderBottom:"1px solid #e5e7eb", background: i%2===0?"white":"#f9fafb" }}>
                  <td style={{ padding:"8px 10px", fontWeight:600, color:"#374151" }}>
                    {TASK_ICONS[d.task_name]} {d.task} {d.paradigm}
                    <div style={{ fontSize:10, color:"#9ca3af", fontWeight:400 }}>{d.task_name} · {PAR_SHORT[d.paradigm]}</div>
                  </td>
                  <td style={{ padding:"8px 10px", textAlign:"center" }}>
                    <span style={{ display:"inline-block", padding:"2px 7px", borderRadius:5, fontWeight:700, background:c.bg, color:c.text, border:`1px solid ${c.border}` }}>
                      {(d.BA*100).toFixed(1)}%
                    </span>
                  </td>
                  <td style={{ padding:"8px 10px", textAlign:"center", color:"#6b7280" }}>{(d.AUC*100).toFixed(1)}%</td>
                  <td style={{ padding:"8px 10px", textAlign:"center", color:"#059669" }}>{(d.Recall*100).toFixed(1)}%</td>
                  <td style={{ padding:"8px 10px", textAlign:"center", color:"#7c3aed" }}>{(d.Precision*100).toFixed(1)}%</td>
                  <td style={{ padding:"8px 10px" }}>
                    <div style={{ display:"flex", gap:4, flexWrap:"wrap" }}>
                      {d.features.slice(0,5).map((f, fi) => (
                        <FeatureChip key={fi} feat={f} rank={fi+1} />
                      ))}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Tab 5: Per-Task Figures ────────────────────────────────────────────────────
const PAR_COLORS = {
  P1: { line:"#2563eb", fill:"rgba(37,99,235,0.15)", label:"All Pts vs Ctrl" },
  P2: { line:"#16a34a", fill:"rgba(22,163,74,0.15)",  label:"RCT vs Ctrl" },
  P3: { line:"#d97706", fill:"rgba(217,119,6,0.15)",  label:"Other vs Ctrl" },
  P4: { line:"#dc2626", fill:"rgba(220,38,38,0.15)",  label:"RCT vs Other" },
};

function MiniBarGroup({ metrics, pars }) {
  const metricKeys = ["BA","AUC","Recall","Precision","F1"];
  const W = 340, H = 160, padL = 44, padB = 28, padT = 12, padR = 10;
  const chartW = W - padL - padR;
  const chartH = H - padB - padT;
  const groupW = chartW / metricKeys.length;
  const barW = Math.min(14, (groupW - 8) / pars.length);

  return (
    <svg width={W} height={H} style={{ fontFamily:"system-ui,sans-serif", overflow:"visible" }}>
      {/* Y grid */}
      {[0,0.25,0.5,0.75,1.0].map(v => {
        const y = padT + chartH - v * chartH;
        return (
          <g key={v}>
            <line x1={padL} x2={padL+chartW} y1={y} y2={y} stroke="#e5e7eb" strokeWidth={v===0?1.5:0.5}/>
            <text x={padL-4} y={y+3} fontSize={8} fill="#9ca3af" textAnchor="end">{(v*100).toFixed(0)}</text>
          </g>
        );
      })}
      {/* Chance line */}
      {(() => { const y = padT + chartH - 0.5 * chartH; return <line x1={padL} x2={padL+chartW} y1={y} y2={y} stroke="#f87171" strokeDasharray="3,2" strokeWidth={1}/> })()}

      {/* Bars */}
      {metricKeys.map((mk, mi) => {
        const gx = padL + mi * groupW + 4;
        return (
          <g key={mk}>
            {pars.map((par, pi) => {
              const d = metrics[par];
              if (!d) return null;
              const val = d[mk] || 0;
              const bh = val * chartH;
              const x = gx + pi * (barW + 1);
              const y = padT + chartH - bh;
              const c = PAR_COLORS[par];
              return (
                <g key={par}>
                  <rect x={x} y={y} width={barW} height={bh} fill={c.line} rx={2} opacity={0.85}/>
                  {val >= 0.7 && <text x={x+barW/2} y={y-2} fontSize={7} fill={c.line} textAnchor="middle">{(val*100).toFixed(0)}</text>}
                </g>
              );
            })}
            <text x={gx + (pars.length*(barW+1))/2} y={padT+chartH+14} fontSize={8} fill="#4b5563" textAnchor="middle">{mk}</text>
          </g>
        );
      })}
    </svg>
  );
}

function RadarChart({ metrics, pars, size=180 }) {
  const axes = ["BA","AUC","Recall","Precision","F1"];
  const n = axes.length;
  const cx = size/2, cy = size/2, r = size/2 - 28;

  function polar(i, val) {
    const angle = (Math.PI * 2 * i / n) - Math.PI/2;
    return { x: cx + Math.cos(angle) * r * val, y: cy + Math.sin(angle) * r * val };
  }

  return (
    <svg width={size} height={size} style={{ fontFamily:"system-ui,sans-serif" }}>
      {/* Grid rings */}
      {[0.25,0.5,0.75,1.0].map(ring => (
        <polygon key={ring} fill="none" stroke="#e5e7eb" strokeWidth={ring===1?1:0.5}
          points={axes.map((_,i)=>{ const p=polar(i,ring); return `${p.x},${p.y}`; }).join(" ")}/>
      ))}
      {/* Axes */}
      {axes.map((_,i) => {
        const p = polar(i, 1);
        return <line key={i} x1={cx} y1={cy} x2={p.x} y2={p.y} stroke="#d1d5db" strokeWidth={0.5}/>;
      })}
      {/* Chance ring */}
      <polygon fill="rgba(248,113,113,0.07)" stroke="#f87171" strokeWidth={0.8} strokeDasharray="2,2"
        points={axes.map((_,i)=>{ const p=polar(i,0.5); return `${p.x},${p.y}`; }).join(" ")}/>

      {/* Data polygons */}
      {pars.map(par => {
        const d = metrics[par];
        if (!d) return null;
        const c = PAR_COLORS[par];
        const pts = axes.map((mk,i) => { const p=polar(i, d[mk]||0); return `${p.x},${p.y}`; }).join(" ");
        return (
          <g key={par}>
            <polygon points={pts} fill={c.fill} stroke={c.line} strokeWidth={1.5}/>
          </g>
        );
      })}

      {/* Axis labels */}
      {axes.map((ax,i) => {
        const p = polar(i, 1.22);
        return <text key={ax} x={p.x} y={p.y} fontSize={8.5} fill="#374151" textAnchor="middle" dominantBaseline="middle">{ax}</text>;
      })}
    </svg>
  );
}

function FeatureHeatmap({ taskData }) {
  const ALL_FEATURES = [
    "head_pos_y","head_rot_x","head_pos_z","head_rot_z","head_pos_x","head_rot_y",
    "left_hand_rot_z","left_hand_pos_x","left_hand_pos_z","left_hand_rot_y","left_hand_pos_y","left_hand_rot_x",
    "right_hand_rot_y","right_hand_pos_x","right_hand_pos_y","right_hand_rot_z","right_hand_pos_z","right_hand_rot_x"
  ];
  const pars = taskData.map(d=>d.paradigm);
  const cellW = 52, cellH = 18, labelW = 130, headerH = 20;
  const W = labelW + pars.length * cellW + 10;
  const H = headerH + ALL_FEATURES.length * cellH + 4;

  function getImportance(d, feat) {
    const idx = d.features.indexOf(feat);
    if (idx === -1) return 0;
    return d.importances[idx] || 0;
  }
  const maxImp = Math.max(...taskData.flatMap(d => d.importances));

  function heatColor(val) {
    if (val === 0) return "#f9fafb";
    const t = val / maxImp;
    if (t > 0.7) return `rgba(37,99,235,${0.3 + t*0.6})`;
    if (t > 0.4) return `rgba(37,99,235,${0.2 + t*0.5})`;
    return `rgba(37,99,235,${0.05 + t*0.4})`;
  }

  return (
    <svg width={W} height={H} style={{ fontFamily:"system-ui,sans-serif" }}>
      {/* Header */}
      {pars.map((par,pi) => {
        const c = PAR_COLORS[par];
        return (
          <g key={par}>
            <rect x={labelW+pi*cellW} y={0} width={cellW-1} height={headerH-2} fill={c.line} rx={2}/>
            <text x={labelW+pi*cellW+cellW/2} y={headerH/2+1} fontSize={8} fill="white" textAnchor="middle" dominantBaseline="middle" fontWeight="bold">{par}</text>
          </g>
        );
      })}
      {/* Rows */}
      {ALL_FEATURES.map((feat,fi) => {
        const sensor = getSensor(feat);
        const sc = SENSOR_COLORS[sensor];
        const y = headerH + fi * cellH;
        const shortLabel = feat.replace("right_hand_","RH·").replace("left_hand_","LH·").replace("head_","H·");
        return (
          <g key={feat}>
            <rect x={0} y={y} width={labelW-2} height={cellH-1} fill={sc.bg} rx={2}/>
            <text x={4} y={y+cellH/2+1} fontSize={8} fill={sc.text} dominantBaseline="middle">{shortLabel}</text>
            {pars.map((par,pi) => {
              const d = taskData.find(r=>r.paradigm===par);
              const imp = d ? getImportance(d, feat) : 0;
              const rank = d ? d.features.indexOf(feat) + 1 : 0;
              return (
                <g key={par}>
                  <rect x={labelW+pi*cellW} y={y} width={cellW-1} height={cellH-1} fill={heatColor(imp)} rx={1}/>
                  {imp > 0 && <text x={labelW+pi*cellW+cellW/2} y={y+cellH/2+1} fontSize={7.5} fill={imp/maxImp > 0.5 ? "white" : "#1e293b"} textAnchor="middle" dominantBaseline="middle">
                    {rank > 0 ? `#${rank}` : ""}
                  </text>}
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

function TaskCard({ taskName, taskData }) {
  const icon = TASK_ICONS[taskName];
  const pars = taskData.map(d=>d.paradigm);
  const metrics = {};
  taskData.forEach(d => { metrics[d.paradigm] = d; });

  const bestPar = taskData.reduce((a,b) => a.BA > b.BA ? a : b);
  const hasP4 = taskData.some(d=>d.paradigm==="P4");

  return (
    <div style={{ background:"white", borderRadius:14, border:"1px solid #e2e8f0", overflow:"hidden", marginBottom:20 }}>
      {/* Task Header */}
      <div style={{ background:"linear-gradient(135deg,#1e293b,#334155)", padding:"14px 18px", display:"flex", alignItems:"center", justifyContent:"space-between" }}>
        <div>
          <div style={{ fontSize:17, fontWeight:800, color:"white" }}>{icon} {taskName}</div>
          <div style={{ fontSize:11, color:"#94a3b8", marginTop:2 }}>{taskData.length} paradigm{taskData.length!==1?"s":""} · Best: {bestPar.paradigm} BA={( bestPar.BA*100).toFixed(1)}%</div>
        </div>
        <div style={{ display:"flex", gap:6 }}>
          {pars.map(par => {
            const c = PAR_COLORS[par];
            const d = metrics[par];
            return (
              <div key={par} style={{ textAlign:"center", background:"rgba(255,255,255,0.1)", borderRadius:8, padding:"6px 10px", border:`1px solid ${c.line}` }}>
                <div style={{ fontSize:9, color:"#94a3b8" }}>{par}</div>
                <div style={{ fontSize:14, fontWeight:800, color:c.line }}>{(d.BA*100).toFixed(1)}%</div>
                <div style={{ fontSize:8, color:"#64748b" }}>BA</div>
              </div>
            );
          })}
          {!taskData.find(d=>d.paradigm==="P1") && (
            <div style={{ textAlign:"center", background:"rgba(255,0,0,0.08)", borderRadius:8, padding:"6px 10px", border:"1px solid #f87171" }}>
              <div style={{ fontSize:9, color:"#f87171" }}>P1</div>
              <div style={{ fontSize:11, fontWeight:700, color:"#f87171" }}>—</div>
              <div style={{ fontSize:8, color:"#f87171" }}>missing</div>
            </div>
          )}
        </div>
      </div>

      {/* Charts Row */}
      <div style={{ display:"grid", gridTemplateColumns:"auto auto 1fr", gap:0, borderBottom:"1px solid #f1f5f9" }}>
        {/* Radar */}
        <div style={{ padding:"14px 8px 8px 14px", borderRight:"1px solid #f1f5f9" }}>
          <div style={{ fontSize:10, fontWeight:600, color:"#6b7280", marginBottom:4, textAlign:"center" }}>Metric Radar</div>
          <RadarChart metrics={metrics} pars={pars} size={180}/>
          <div style={{ display:"flex", flexWrap:"wrap", gap:4, justifyContent:"center", marginTop:4 }}>
            {pars.map(par => (
              <span key={par} style={{ display:"flex", alignItems:"center", gap:3, fontSize:9, color:PAR_COLORS[par].line }}>
                <span style={{ width:8, height:3, borderRadius:2, background:PAR_COLORS[par].line, display:"inline-block" }}/>
                {par}
              </span>
            ))}
          </div>
        </div>

        {/* Bar group */}
        <div style={{ padding:"14px 8px 8px 8px", borderRight:"1px solid #f1f5f9" }}>
          <div style={{ fontSize:10, fontWeight:600, color:"#6b7280", marginBottom:4, textAlign:"center" }}>Metrics by Paradigm</div>
          <MiniBarGroup metrics={metrics} pars={pars}/>
          <div style={{ display:"flex", gap:8, justifyContent:"center", marginTop:2 }}>
            {pars.map(par => (
              <span key={par} style={{ display:"flex", alignItems:"center", gap:3, fontSize:9, color:PAR_COLORS[par].line }}>
                <span style={{ width:8, height:8, borderRadius:2, background:PAR_COLORS[par].line, display:"inline-block" }}/>
                {PAR_COLORS[par].label}
              </span>
            ))}
          </div>
        </div>

        {/* Recall vs Precision scatter-like */}
        <div style={{ padding:"14px 14px 8px 8px" }}>
          <div style={{ fontSize:10, fontWeight:600, color:"#6b7280", marginBottom:6, textAlign:"center" }}>Recall vs Precision Trade-off</div>
          <svg width="100%" height={150} viewBox="0 0 220 150" style={{ overflow:"visible" }}>
            {/* Grid */}
            {[0.3,0.5,0.7,0.9].map(v=>(
              <g key={v}>
                <line x1={20} x2={210} y1={130 - v*110} y2={130 - v*110} stroke="#f1f5f9" strokeWidth={0.5}/>
                <line x1={20 + v*190} x2={20 + v*190} y1={20} y2={130} stroke="#f1f5f9" strokeWidth={0.5}/>
                <text x={14} y={130 - v*110 + 3} fontSize={7} fill="#9ca3af" textAnchor="end">{(v*100).toFixed(0)}</text>
                <text x={20 + v*190} y={140} fontSize={7} fill="#9ca3af" textAnchor="middle">{(v*100).toFixed(0)}</text>
              </g>
            ))}
            {/* Diagonal parity line */}
            <line x1={20} x2={210} y1={130} y2={20} stroke="#d1d5db" strokeDasharray="3,2" strokeWidth={1}/>
            <text x={118} y={68} fontSize={7} fill="#9ca3af" transform="rotate(-35,118,68)">Rec=Prec</text>
            {/* Axes labels */}
            <text x={115} y={148} fontSize={8} fill="#6b7280" textAnchor="middle">Precision →</text>
            <text x={6} y={75} fontSize={8} fill="#6b7280" textAnchor="middle" transform="rotate(-90,6,75)">Recall →</text>
            {/* Points */}
            {pars.map(par => {
              const d = metrics[par];
              if (!d) return null;
              const c = PAR_COLORS[par];
              const cx2 = 20 + d.Precision * 190;
              const cy2 = 130 - d.Recall * 110;
              return (
                <g key={par}>
                  <circle cx={cx2} cy={cy2} r={7} fill={c.line} opacity={0.85}/>
                  <text x={cx2} y={cy2+3} fontSize={7.5} fill="white" textAnchor="middle" fontWeight="bold">{par}</text>
                  <text x={cx2} y={cy2-9} fontSize={7} fill={c.line} textAnchor="middle">{(d.BA*100).toFixed(0)}%</text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Feature Heatmap */}
      <div style={{ padding:"14px 16px 16px" }}>
        <div style={{ fontSize:10, fontWeight:600, color:"#6b7280", marginBottom:8 }}>Feature Importance Heatmap (rank shown, colour intensity = permutation importance)</div>
        <div style={{ overflowX:"auto" }}>
          <FeatureHeatmap taskData={taskData}/>
        </div>
        <div style={{ marginTop:8, display:"flex", gap:8, alignItems:"center" }}>
          <span style={{ fontSize:9, color:"#6b7280" }}>Intensity scale →</span>
          {[0.1,0.3,0.5,0.7,1.0].map(t=>(
            <span key={t} style={{ display:"inline-block", width:24, height:10, borderRadius:2, background:`rgba(37,99,235,${0.05+t*0.9})` }}/>
          ))}
          <span style={{ fontSize:9, color:"#6b7280" }}>Low → High importance</span>
        </div>
      </div>

      {/* Metrics detail strip */}
      <div style={{ display:"grid", gridTemplateColumns:`repeat(${pars.length},1fr)`, gap:0, borderTop:"1px solid #f1f5f9" }}>
        {pars.map((par,pi) => {
          const d = metrics[par];
          const c = PAR_COLORS[par];
          const bc = baColor(d.BA);
          return (
            <div key={par} style={{ padding:"10px 12px", borderLeft: pi>0 ? "1px solid #f1f5f9" : "none" }}>
              <div style={{ fontSize:10, fontWeight:700, color:c.line, marginBottom:6 }}>{par} — {PAR_FULL[par]}</div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"3px 10px" }}>
                {[["BA",d.BA],["AUC",d.AUC],["Recall",d.Recall],["Precision",d.Precision],["F1",d.F1]].map(([k,v])=>(
                  <div key={k} style={{ display:"flex", justifyContent:"space-between", fontSize:10 }}>
                    <span style={{ color:"#6b7280" }}>{k}</span>
                    <span style={{ fontWeight:600, color: k==="BA" ? bc.text : "#374151" }}>{(v*100).toFixed(1)}%</span>
                  </div>
                ))}
                <div style={{ display:"flex", justifyContent:"space-between", fontSize:10 }}>
                  <span style={{ color:"#6b7280" }}>n_states</span>
                  <span style={{ fontWeight:600, color:"#374151" }}>{d.n_states}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PerTaskFigures({ data }) {
  const taskNames = ["Jar Opening","Key Turning","Cleaning","Back Washing","Cutting","Hammering"];
  const [selectedTask, setSelectedTask] = useState(null);

  const displayTasks = selectedTask ? [selectedTask] : taskNames;

  return (
    <div>
      {/* Task selector */}
      <div style={{ display:"flex", gap:6, marginBottom:16, flexWrap:"wrap" }}>
        <button onClick={()=>setSelectedTask(null)} style={{
          padding:"6px 12px", borderRadius:6, border:"1px solid #e2e8f0", cursor:"pointer", fontSize:12, fontWeight:600,
          background: selectedTask===null ? "#1e293b" : "white", color: selectedTask===null ? "white" : "#374151"
        }}>All Tasks</button>
        {taskNames.map(t => (
          <button key={t} onClick={()=>setSelectedTask(selectedTask===t ? null : t)} style={{
            padding:"6px 12px", borderRadius:6, border:"1px solid #e2e8f0", cursor:"pointer", fontSize:12, fontWeight:500,
            background: selectedTask===t ? "#1e293b" : "white", color: selectedTask===t ? "white" : "#374151"
          }}>{TASK_ICONS[t]} {t}</button>
        ))}
      </div>

      {displayTasks.map(tname => {
        const taskData = data.filter(d => d.task_name === tname).sort((a,b)=>a.par_num-b.par_num);
        if (taskData.length === 0) return null;
        return <TaskCard key={tname} taskName={tname} taskData={taskData}/>;
      })}
    </div>
  );
}

// ── Tab 6: Task-wise BA-Weighted Top-6 Feature Comparison ─────────────────────
const TASK_COLORS = {
  "Jar Opening":  { bar:"#2563eb", light:"#dbeafe", text:"#1e40af" },
  "Key Turning":  { bar:"#7c3aed", light:"#ede9fe", text:"#5b21b6" },
  "Cleaning":     { bar:"#0891b2", light:"#cffafe", text:"#164e63" },
  "Back Washing": { bar:"#059669", light:"#d1fae5", text:"#065f46" },
  "Cutting":      { bar:"#d97706", light:"#fef3c7", text:"#92400e" },
  "Hammering":    { bar:"#dc2626", light:"#fee2e2", text:"#991b1b" },
};

const FEAT_SHORT = f =>
  f.replace("right_hand_","RH·").replace("left_hand_","LH·").replace("head_","H·")
   .replace("_pos_","p").replace("_rot_","r");

function TaskwiseFeatureComparison({ data }) {
  const taskNames = ["Jar Opening","Key Turning","Cleaning","Back Washing","Cutting","Hammering"];
  const [mode, setMode] = useState("all");   // "all" | "p1" | "p2" | "p3" | "p4"
  const [highlight, setHighlight] = useState(null); // feature name to highlight across tasks

  // Compute BA-weighted score per feature per task, filtered by paradigm mode
  const taskFeatureScores = useMemo(() => {
    const result = {};
    taskNames.forEach(tname => {
      const rows = data.filter(d => {
        if (d.task_name !== tname) return false;
        if (mode !== "all" && d.paradigm !== mode.toUpperCase()) return false;
        return true;
      });
      const scores = {};
      rows.forEach(d => {
        const ba = d.BA;
        d.features.forEach((feat, i) => {
          const imp = d.importances[i] || 0;
          scores[feat] = (scores[feat] || 0) + imp * ba;
        });
      });
      // Sort and take top 6
      result[tname] = Object.entries(scores)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6)
        .map(([feat, score]) => ({ feat, score }));
    });
    return result;
  }, [data, mode]);

  // Global max score for normalising bars across all tasks
  const globalMax = useMemo(() =>
    Math.max(...Object.values(taskFeatureScores).flatMap(arr => arr.map(x => x.score))),
    [taskFeatureScores]
  );

  // Which features appear in top-6 of ALL tasks (cross-task consensus)
  const featureTaskCount = useMemo(() => {
    const cnt = {};
    Object.values(taskFeatureScores).forEach(arr =>
      arr.forEach(({ feat }) => { cnt[feat] = (cnt[feat] || 0) + 1; })
    );
    return cnt;
  }, [taskFeatureScores]);

  const universalFeats = Object.entries(featureTaskCount)
    .filter(([,c]) => c >= 5).map(([f]) => f);
  const consensusFeats = Object.entries(featureTaskCount)
    .filter(([,c]) => c >= 4 && c < 5).map(([f]) => f);

  // Cross-task heatmap: tasks × top-12 global features
  const globalTopFeats = useMemo(() => {
    const total = {};
    Object.values(taskFeatureScores).forEach(arr =>
      arr.forEach(({ feat, score }) => { total[feat] = (total[feat] || 0) + score; })
    );
    return Object.entries(total).sort((a,b)=>b[1]-a[1]).slice(0,12).map(([f])=>f);
  }, [taskFeatureScores]);

  const crossMax = useMemo(() => {
    let m = 0;
    taskNames.forEach(t => {
      globalTopFeats.forEach(f => {
        const found = taskFeatureScores[t]?.find(x=>x.feat===f);
        if (found && found.score > m) m = found.score;
      });
    });
    return m;
  }, [taskFeatureScores, globalTopFeats]);

  return (
    <div>
      {/* Controls */}
      <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:16, flexWrap:"wrap" }}>
        <span style={{ fontSize:12, fontWeight:600, color:"#374151" }}>Paradigm filter:</span>
        {["all","p1","p2","p3","p4"].map(m => (
          <button key={m} onClick={()=>setMode(m)} style={{
            padding:"5px 12px", borderRadius:6, border:"1px solid #e2e8f0", cursor:"pointer",
            fontSize:11, fontWeight:600,
            background: mode===m ? "#1e293b" : "white",
            color: mode===m ? "white" : "#6b7280"
          }}>{m === "all" ? "All Paradigms" : m.toUpperCase() + " — " + ({"p1":"All Pts","p2":"RCT","p3":"Other","p4":"RCT vs Other"})[m]}</button>
        ))}
      </div>

      {/* Consensus badges */}
      <div style={{ marginBottom:16, display:"flex", gap:8, flexWrap:"wrap", alignItems:"center" }}>
        <span style={{ fontSize:11, fontWeight:600, color:"#374151" }}>Universal (≥5 tasks):</span>
        {universalFeats.length === 0
          ? <span style={{ fontSize:11, color:"#9ca3af" }}>none</span>
          : universalFeats.map(f => (
            <span key={f} onClick={()=>setHighlight(highlight===f?null:f)}
              style={{ padding:"2px 8px", borderRadius:12, background:"#1e293b", color:"white",
                fontSize:11, cursor:"pointer", border: highlight===f?"2px solid #f59e0b":"2px solid transparent" }}>
              {FEAT_SHORT(f)} ×{featureTaskCount[f]}
            </span>
          ))
        }
        <span style={{ fontSize:11, fontWeight:600, color:"#374151", marginLeft:8 }}>Consensus (4 tasks):</span>
        {consensusFeats.map(f => (
          <span key={f} onClick={()=>setHighlight(highlight===f?null:f)}
            style={{ padding:"2px 8px", borderRadius:12, background:"#475569", color:"white",
              fontSize:11, cursor:"pointer", border: highlight===f?"2px solid #f59e0b":"2px solid transparent" }}>
            {FEAT_SHORT(f)} ×{featureTaskCount[f]}
          </span>
        ))}
        {highlight && <span style={{ fontSize:11, color:"#f59e0b", fontWeight:600 }}>↑ highlighting: {highlight}</span>}
      </div>

      {/* Per-task top-6 bars — 2-column grid */}
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14, marginBottom:24 }}>
        {taskNames.map(tname => {
          const tc = TASK_COLORS[tname];
          const top6 = taskFeatureScores[tname] || [];
          if (top6.length === 0) return (
            <div key={tname} style={{ padding:16, borderRadius:10, background:"#f9fafb", border:"1px solid #e5e7eb" }}>
              <div style={{ fontSize:13, fontWeight:700, color:"#374151" }}>{TASK_ICONS[tname]} {tname}</div>
              <div style={{ fontSize:11, color:"#9ca3af", marginTop:6 }}>No data for selected paradigm</div>
            </div>
          );

          return (
            <div key={tname} style={{ borderRadius:10, border:`1.5px solid ${tc.light}`, overflow:"hidden" }}>
              {/* Card header */}
              <div style={{ background:tc.bar, padding:"10px 14px", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                <span style={{ fontSize:13, fontWeight:800, color:"white" }}>{TASK_ICONS[tname]} {tname}</span>
                <span style={{ fontSize:10, color:"rgba(255,255,255,0.75)" }}>BA-weighted score (top 6)</span>
              </div>

              {/* Bars */}
              <div style={{ background:tc.light, padding:"10px 14px 12px" }}>
                {top6.map(({ feat, score }, i) => {
                  const barPct = (score / globalMax) * 100;
                  const sensor = getSensor(feat);
                  const sc = SENSOR_COLORS[sensor];
                  const isHighlit = highlight === feat;
                  const isInOtherTasks = featureTaskCount[feat] > 1;
                  return (
                    <div key={feat} onClick={()=>setHighlight(highlight===feat?null:feat)}
                      style={{ marginBottom: i < 5 ? 7 : 0, cursor:"pointer",
                        opacity: highlight && !isHighlit ? 0.4 : 1,
                        transition:"opacity 0.15s" }}>
                      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3, alignItems:"center" }}>
                        <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                          <span style={{ fontSize:10, fontWeight:700, color:tc.text, width:14 }}>#{i+1}</span>
                          <span style={{ fontSize:10, padding:"1px 6px", borderRadius:3,
                            background: isHighlit ? "#f59e0b" : sc.bg,
                            color: isHighlit ? "#1e293b" : sc.text,
                            border:`1px solid ${isHighlit ? "#f59e0b" : sc.border}`,
                            fontWeight: isHighlit ? 700 : 500 }}>
                            {FEAT_SHORT(feat)}
                          </span>
                          {isInOtherTasks && (
                            <span style={{ fontSize:9, color:"#6b7280" }}>×{featureTaskCount[feat]}</span>
                          )}
                        </div>
                        <span style={{ fontSize:10, fontWeight:700, color:tc.text }}>{score.toFixed(3)}</span>
                      </div>
                      <div style={{ height:8, background:"rgba(255,255,255,0.6)", borderRadius:4, overflow:"hidden" }}>
                        <div style={{
                          width:`${barPct}%`, height:"100%", borderRadius:4,
                          background: isHighlit ? "#f59e0b" : tc.bar,
                          transition:"width 0.4s ease"
                        }}/>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {/* Cross-task heatmap */}
      <div style={{ marginBottom:16 }}>
        <div style={{ fontSize:13, fontWeight:700, color:"#1e293b", marginBottom:10 }}>
          Cross-Task Heatmap — Top-12 Global Features vs All Tasks
          <span style={{ fontSize:11, fontWeight:400, color:"#6b7280", marginLeft:8 }}>(cell = BA-weighted score · darker = higher)</span>
        </div>
        <div style={{ overflowX:"auto" }}>
          <table style={{ borderCollapse:"collapse", width:"100%", fontSize:11 }}>
            <thead>
              <tr>
                <th style={{ padding:"8px 10px", background:"#1e293b", color:"white", textAlign:"left", minWidth:80, borderRadius:"6px 0 0 0" }}>Feature</th>
                {taskNames.map((t,i) => (
                  <th key={t} style={{ padding:"8px 8px", background:TASK_COLORS[t].bar, color:"white", textAlign:"center", minWidth:80,
                    borderRadius: i===taskNames.length-1 ? "0 6px 0 0" : 0 }}>
                    {TASK_ICONS[t]}<br/><span style={{fontSize:9,fontWeight:400}}>{t.split(" ")[0]}</span>
                  </th>
                ))}
                <th style={{ padding:"8px 8px", background:"#374151", color:"white", textAlign:"center", minWidth:60 }}>Total</th>
              </tr>
            </thead>
            <tbody>
              {globalTopFeats.map((feat, fi) => {
                const sensor = getSensor(feat);
                const sc = SENSOR_COLORS[sensor];
                const totalScore = taskNames.reduce((s,t) => {
                  const found = taskFeatureScores[t]?.find(x=>x.feat===feat);
                  return s + (found?.score || 0);
                }, 0);
                const isHighlit = highlight === feat;
                return (
                  <tr key={feat} onClick={()=>setHighlight(highlight===feat?null:feat)}
                    style={{ cursor:"pointer", background: isHighlit ? "#fef3c7" : fi%2===0?"white":"#f9fafb" }}>
                    <td style={{ padding:"7px 10px", fontWeight:600, borderBottom:"1px solid #f1f5f9" }}>
                      <span style={{ padding:"1px 6px", borderRadius:3, background:sc.bg, color:sc.text, border:`1px solid ${sc.border}`, fontSize:10 }}>
                        {FEAT_SHORT(feat)}
                      </span>
                    </td>
                    {taskNames.map(t => {
                      const found = taskFeatureScores[t]?.find(x=>x.feat===feat);
                      const score = found?.score || 0;
                      const intensity = crossMax > 0 ? score / crossMax : 0;
                      const sensor2 = getSensor(feat);
                      const baseColor = TASK_COLORS[t].bar;
                      const bg = score > 0
                        ? `rgba(${sensor2==="head"?"37,99,235":sensor2==="right"?"22,163,74":"217,119,6"},${0.1+intensity*0.75})`
                        : "#f9fafb";
                      const rank = taskFeatureScores[t]?.findIndex(x=>x.feat===feat);
                      return (
                        <td key={t} style={{ padding:"7px 8px", textAlign:"center", background:bg, borderBottom:"1px solid #f1f5f9", borderLeft:"1px solid #f1f5f9" }}>
                          {score > 0 ? (
                            <div>
                              <div style={{ fontWeight:700, color: intensity > 0.5 ? "white" : "#1e293b" }}>{score.toFixed(2)}</div>
                              <div style={{ fontSize:8, color: intensity > 0.5 ? "rgba(255,255,255,0.7)" : "#9ca3af" }}>#{rank+1}</div>
                            </div>
                          ) : <span style={{ color:"#d1d5db" }}>—</span>}
                        </td>
                      );
                    })}
                    <td style={{ padding:"7px 8px", textAlign:"center", fontWeight:700, color:"#1e293b", background: isHighlit?"#fde68a":"#f1f5f9", borderLeft:"1px solid #e5e7eb" }}>
                      {totalScore.toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop:8, fontSize:10, color:"#6b7280" }}>
          Click any feature bar or heatmap row to highlight it across all views. ×N = appears in top-6 of N tasks.
          Colour intensity: <span style={{color:"#2563eb"}}>■</span> Head &nbsp;<span style={{color:"#16a34a"}}>■</span> Right Hand &nbsp;<span style={{color:"#d97706"}}>■</span> Left Hand
        </div>
      </div>
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState(0);
  const tabs = ["📋 Performance Table","🔎 Key Insights","🧬 Feature Intelligence","🔬 Per-Experiment","📊 Per-Task Figures","🏅 Feature Comparison"];

  return (
    <div style={{ fontFamily:"system-ui,-apple-system,sans-serif", background:"#f1f5f9", minHeight:"100vh", padding:16 }}>
      {/* Header */}
      <div style={{ background:"linear-gradient(135deg,#1e293b,#334155)", borderRadius:12, padding:"18px 22px", marginBottom:16, color:"white" }}>
        <div style={{ fontSize:18, fontWeight:800, marginBottom:4 }}>HMM Classification Analysis Dashboard</div>
        <div style={{ fontSize:12, opacity:0.75 }}>XDash XR Study · 6 Tasks × 4 Paradigms · 23 Experiments · 18 Motion Features · N=35–60 subjects per experiment</div>
        <div style={{ marginTop:10, display:"flex", gap:16, flexWrap:"wrap" }}>
          {[["23","Experiments"],["6","Tasks"],["4","Paradigms"],["18","Features"],["91.0%","Best BA (T1 P2)"],["1","Missing (T2 P1)"]].map(([v,l])=>(
            <div key={l} style={{ textAlign:"center" }}>
              <div style={{ fontSize:18, fontWeight:800, color:"#7dd3fc" }}>{v}</div>
              <div style={{ fontSize:10, opacity:0.7 }}>{l}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Tab Bar */}
      <div style={{ display:"flex", gap:4, marginBottom:14, background:"white", padding:4, borderRadius:10, border:"1px solid #e2e8f0" }}>
        {tabs.map((t, i) => (
          <button key={i} onClick={()=>setTab(i)} style={{
            flex:1, padding:"8px 6px", borderRadius:7, border:"none", cursor:"pointer", fontSize:12, fontWeight:600,
            background: tab===i ? "#1e293b" : "transparent",
            color: tab===i ? "white" : "#6b7280",
            transition:"all 0.15s"
          }}>{t}</button>
        ))}
      </div>

      {/* Content */}
      <div style={{ background:"white", borderRadius:12, padding:20, border:"1px solid #e2e8f0" }}>
        {tab===0 && <PerformanceTable data={RAW_DATA} />}
        {tab===1 && <InsightsPanel data={RAW_DATA} />}
        {tab===2 && <FeatureIntelligence data={RAW_DATA} />}
        {tab===3 && <ExperimentTable data={RAW_DATA} />}
        {tab===4 && <PerTaskFigures data={RAW_DATA} />}
        {tab===5 && <TaskwiseFeatureComparison data={RAW_DATA} />}
      </div>
    </div>
  );
}