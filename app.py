import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="PipeGuard | Corrosion Risk",page_icon="🧪",layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#f7fbff,#f2fbf8 55%,#fff8ed);color:#243746}
.block-container{max-width:1550px;padding-top:1rem}
[data-testid="stSidebar"]{background:#eff8fb;border-right:1px solid #d9e7ed}
.hero{background:linear-gradient(120deg,#075985,#0284c7,#059669);padding:30px 34px;border-radius:28px;color:white;margin-bottom:18px;box-shadow:0 15px 40px #0284c722}
.hero h1{color:white!important;font-size:2.35rem;margin:8px 0}.hero p{color:#effcff}
.card{background:white;border:1px solid #dce9ee;border-radius:20px;padding:18px;margin-bottom:16px;box-shadow:0 8px 25px #1e465a10}
.title{font-weight:850;color:#164e63;font-size:1.1rem}.muted{color:#718693;font-size:.8rem}
.alert{background:#fff0f0;border-left:4px solid #dc2626;border-radius:10px;padding:10px;margin:6px 0}
.warn{background:#fff7e7;border-left:4px solid #d97706;border-radius:10px;padding:10px;margin:6px 0}
.good{background:#effbf5;border-left:4px solid #059669;border-radius:10px;padding:10px;margin:6px 0}
div[data-testid="stMetric"]{background:white;border:1px solid #dce9ee;border-radius:15px;padding:10px}
.footer{text-align:center;color:#78909c;padding:20px;font-size:.75rem}
</style>""",unsafe_allow_html=True)

DATA=Path(__file__).parent/"data"/"sample_pipeline_corrosion_records.csv"
REQ=["pipe_id","material","installation_year","age_years","soil_type","soil_corrosivity","ph","chloride_mg_l","temperature_c","pressure_bar","flow_m3_day","repair_count_5y","leak_reports_3y","last_inspection","coating_condition","cathodic_protection","zone","latitude","longitude"]
@st.cache_data
def load(): return pd.read_csv(DATA)
def prep(d):
 x=d.copy()
 nums=["installation_year","age_years","soil_corrosivity","ph","chloride_mg_l","temperature_c","pressure_bar","flow_m3_day","repair_count_5y","leak_reports_3y","latitude","longitude"]
 for c in nums:x[c]=pd.to_numeric(x[c],errors="coerce").fillna(0)
 x["last_inspection"]=pd.to_datetime(x["last_inspection"],errors="coerce")
 mat=x.material.map({"PVC":10,"HDPE":8,"Ductile Iron":35,"Cast Iron":65,"Steel":55,"Asbestos Cement":60}).fillna(45)
 chem=np.clip(.55*abs(x.ph-7)/3*100+.45*x.chloride_mg_l/400*100,0,100)
 age=np.clip(x.age_years/80*100,0,100); soil=np.clip(x.soil_corrosivity,0,100)
 pressure=np.clip(x.pressure_bar/12*100,0,100); repairs=np.clip(x.repair_count_5y/5*100,0,100)
 leaks=np.clip(x.leak_reports_3y/4*100,0,100); temp=np.clip((x.temperature_c-5)/40*100,0,100)
 coat=x.coating_condition.map({"Excellent":10,"Good":25,"Fair":55,"Poor":85,"Unknown":55}).fillna(55)
 cath=x.cathodic_protection.map({"Active":10,"Monitored":25,"Inactive":75,"None":90,"Unknown":55}).fillna(55)
 x["corrosion_score"]=np.round(np.clip(.20*age+.17*soil+.16*chem+.13*mat+.10*pressure+.08*repairs+.07*leaks+.04*temp+.03*coat+.02*cath,0,100),1)
 x["leakage_score"]=np.round(np.clip(.35*leaks+.25*repairs+.20*age+.12*pressure+.08*soil,0,100),1)
 x["asset_health"]=np.round(100-x.corrosion_score,1)
 x["risk_band"]=pd.cut(x.corrosion_score,[-1,24,49,74,100],labels=["LOW","MODERATE","HIGH","CRITICAL"]).astype(str)
 x["priority"]=np.select([x.corrosion_score>=75,x.corrosion_score>=50,x.corrosion_score>=25],["URGENT","PRIORITY","MONITOR"],default="ROUTINE")
 return x
df=prep(load())

with st.sidebar:
 st.markdown("## 🧪 PipeGuard")
 st.caption("PIPE ASSET INTELLIGENCE")
 page=st.radio("WORKSPACE",["🏠 Command Center","🧪 Pipe Assessment","🗺️ Network Map","🧱 Material Lab","🌱 Soil & Chemistry","💧 Pressure & Flow","🔧 Repair & Leak Intelligence","🛡️ Protection & Coating","🏙️ Zone Benchmark","📈 Risk Analytics","🚨 Priority Control Tower","📅 Inspection Trends","🧮 What-If Simulator","🔔 Rule-Based Alerts","📂 Data Operations","⚙️ Methodology"])
 st.divider();st.caption("100% LOCAL PROCESSING");st.write("Pandas • NumPy • Plotly");st.write(f"Assets: **{len(df)}**");st.write(f"Zones: **{df.zone.nunique()}**");st.caption("No external APIs.")

st.markdown('<div class="hero"><div>ADVANCED • LOCAL-FIRST • EXPLAINABLE ASSET SCREENING</div><h1>🧪 Water-Pipeline Corrosion Risk Predictor</h1><p>Professional local analytics for potential corrosion and leakage risk using material, age, soil, water chemistry, pressure, protection, coating and repair history.</p></div>',unsafe_allow_html=True)

if page=="🏠 Command Center":
 c=st.columns(5);c[0].metric("Pipe assets",len(df));c[1].metric("Avg corrosion risk",f"{df.corrosion_score.mean():.1f}/100");c[2].metric("High/Critical",int((df.corrosion_score>=50).sum()));c[3].metric("Avg asset health",f"{df.asset_health.mean():.1f}/100");c[4].metric("Leak reports",int(df.leak_reports_3y.sum()))
 a,b=st.columns([1.35,1])
 with a:
  st.markdown('<div class="card"><div class="title">🗺️ Network Risk Landscape</div><div class="muted">Local coordinates colored by screening risk.</div>',unsafe_allow_html=True)
  fig=px.scatter(df,x="longitude",y="latitude",size="flow_m3_day",color="corrosion_score",hover_name="pipe_id",hover_data=["material","zone","age_years"],color_continuous_scale=["#059669","#0284c7","#f59e0b","#dc2626"]);fig.update_layout(height=450);st.plotly_chart(fig,use_container_width=True);st.markdown("</div>",unsafe_allow_html=True)
 with b:
  st.markdown('<div class="card"><div class="title">Risk Portfolio</div>',unsafe_allow_html=True)
  q=df.risk_band.value_counts().reindex(["LOW","MODERATE","HIGH","CRITICAL"]).fillna(0);fig=px.pie(values=q.values,names=q.index,hole=.6,color=q.index,color_discrete_map={"LOW":"#059669","MODERATE":"#0284c7","HIGH":"#f59e0b","CRITICAL":"#dc2626"});fig.update_layout(height=450);st.plotly_chart(fig,use_container_width=True);st.markdown("</div>",unsafe_allow_html=True)
 st.markdown('<div class="card"><div class="title">Priority Pipe Queue</div>',unsafe_allow_html=True)
 for _,r in df.nlargest(7,"corrosion_score").iterrows(): st.markdown(f'<div class="{"alert" if r.corrosion_score>=75 else "warn"}"><b>{r.pipe_id}</b> · {r.risk_band} · {r.material} · {r.zone} · score {r.corrosion_score:.1f}</div>',unsafe_allow_html=True)
 st.markdown("</div>",unsafe_allow_html=True)

elif page=="🧪 Pipe Assessment":
 pid=st.selectbox("Select pipe",df.pipe_id.tolist());r=df[df.pipe_id==pid].iloc[0]
 c=st.columns(5);c[0].metric("Corrosion risk",r.corrosion_score);c[1].metric("Leakage risk",r.leakage_score);c[2].metric("Asset health",r.asset_health);c[3].metric("Age",f"{r.age_years:.0f} yrs");c[4].metric("Repairs / 5y",r.repair_count_5y)
 st.markdown(f'<div class="{"alert" if r.corrosion_score>=75 else "warn" if r.corrosion_score>=50 else "good"}"><b>{r.priority}</b> — {r.risk_band} screening classification.</div>',unsafe_allow_html=True)
 fields=["material","installation_year","age_years","soil_type","soil_corrosivity","ph","chloride_mg_l","temperature_c","pressure_bar","flow_m3_day","repair_count_5y","leak_reports_3y","coating_condition","cathodic_protection","zone","last_inspection"]
 st.dataframe(pd.DataFrame({"Field":fields,"Value":[r[f] for f in fields]}),use_container_width=True,hide_index=True)
 factors={"Age":r.age_years/80*100,"Soil":r.soil_corrosivity,"Chemistry":min(100,abs(r.ph-7)/3*100+r.chloride_mg_l/400*100),"Pressure":min(100,r.pressure_bar/12*100),"Leak history":min(100,r.leak_reports_3y/4*100)}
 fig=px.bar(pd.DataFrame({"Factor":list(factors),"Signal":list(factors.values())}),x="Signal",y="Factor",orientation="h",color="Signal",color_continuous_scale=["#059669","#0284c7","#f59e0b","#dc2626"]);fig.update_layout(height=340);st.plotly_chart(fig,use_container_width=True)

elif page=="🗺️ Network Map":
 metric=st.selectbox("Map metric",["corrosion_score","leakage_score","asset_health","pressure_bar","age_years"])
 fig=px.scatter_mapbox(df,lat="latitude",lon="longitude",color=metric,size="flow_m3_day",hover_name="pipe_id",hover_data=["material","zone","risk_band"],zoom=10,height=600,color_continuous_scale=["#059669","#0284c7","#f59e0b","#dc2626"]);fig.update_layout(mapbox_style="open-street-map",margin=dict(l=0,r=0,t=0,b=0));st.plotly_chart(fig,use_container_width=True)

elif page=="🧱 Material Lab":
 g=df.groupby("material",as_index=False).agg(Pipes=("pipe_id","count"),AvgRisk=("corrosion_score","mean"),AvgAge=("age_years","mean"),Leaks=("leak_reports_3y","sum"));fig=px.bar(g,x="material",y="AvgRisk",color="AvgAge");fig.update_layout(height=430);st.plotly_chart(fig,use_container_width=True);st.dataframe(g,use_container_width=True,hide_index=True)

elif page=="🌱 Soil & Chemistry":
 fig=px.scatter(df,x="soil_corrosivity",y="chloride_mg_l",size="flow_m3_day",color="corrosion_score",hover_name="pipe_id",color_continuous_scale=["#059669","#0284c7","#f59e0b","#dc2626"]);fig.update_layout(height=450);st.plotly_chart(fig,use_container_width=True);st.dataframe(df.nlargest(15,"corrosion_score")[["pipe_id","soil_type","soil_corrosivity","ph","chloride_mg_l","corrosion_score"]],use_container_width=True,hide_index=True)

elif page=="💧 Pressure & Flow":
 fig=px.scatter(df,x="pressure_bar",y="flow_m3_day",size="age_years",color="leakage_score",hover_name="pipe_id");fig.update_layout(height=450);st.plotly_chart(fig,use_container_width=True);st.dataframe(df.nlargest(15,"pressure_bar")[["pipe_id","pressure_bar","flow_m3_day","leak_reports_3y","leakage_score"]],use_container_width=True,hide_index=True)

elif page=="🔧 Repair & Leak Intelligence":
 fig=px.scatter(df,x="repair_count_5y",y="leak_reports_3y",size="age_years",color="corrosion_score",hover_name="pipe_id");fig.update_layout(height=450);st.plotly_chart(fig,use_container_width=True);st.dataframe(df.sort_values(["leak_reports_3y","repair_count_5y"],ascending=False).head(20)[["pipe_id","zone","age_years","repair_count_5y","leak_reports_3y","leakage_score"]],use_container_width=True,hide_index=True)

elif page=="🛡️ Protection & Coating":
 g=df.groupby(["coating_condition","cathodic_protection"],as_index=False).agg(Pipes=("pipe_id","count"),AvgRisk=("corrosion_score","mean"));fig=px.bar(g,x="coating_condition",y="AvgRisk",color="cathodic_protection",barmode="group");fig.update_layout(height=430);st.plotly_chart(fig,use_container_width=True);st.dataframe(g,use_container_width=True,hide_index=True)

elif page=="🏙️ Zone Benchmark":
 g=df.groupby("zone",as_index=False).agg(Pipes=("pipe_id","count"),Risk=("corrosion_score","mean"),Leakage=("leakage_score","mean"),Health=("asset_health","mean"),Leaks=("leak_reports_3y","sum"));fig=px.bar(g,x="zone",y=["Risk","Leakage"],barmode="group");fig.update_layout(height=430);st.plotly_chart(fig,use_container_width=True);st.dataframe(g,use_container_width=True,hide_index=True)

elif page=="📈 Risk Analytics":
 x=st.selectbox("X-axis",["age_years","soil_corrosivity","ph","chloride_mg_l","temperature_c","pressure_bar","flow_m3_day","repair_count_5y","leak_reports_3y"]);y=st.selectbox("Y-axis",["corrosion_score","leakage_score","asset_health"]);fig=px.scatter(df,x=x,y=y,size="flow_m3_day",color="material",hover_name="pipe_id");fig.update_layout(height=470);st.plotly_chart(fig,use_container_width=True)

elif page=="🚨 Priority Control Tower":
 x=df.sort_values("corrosion_score",ascending=False);fig=px.bar(x.head(20),x="corrosion_score",y="pipe_id",orientation="h",color="risk_band",color_discrete_map={"LOW":"#059669","MODERATE":"#0284c7","HIGH":"#f59e0b","CRITICAL":"#dc2626"});fig.update_layout(height=570);st.plotly_chart(fig,use_container_width=True);st.dataframe(x.head(25)[["pipe_id","material","zone","corrosion_score","risk_band","priority"]],use_container_width=True,hide_index=True)

elif page=="📅 Inspection Trends":
 t=df.groupby(df.last_inspection.dt.date).agg(Records=("pipe_id","count"),AvgRisk=("corrosion_score","mean")).reset_index();fig=px.line(t,x="last_inspection",y="AvgRisk",markers=True);fig.update_layout(height=430);st.plotly_chart(fig,use_container_width=True);st.dataframe(t,use_container_width=True,hide_index=True)

elif page=="🧮 What-If Simulator":
 pid=st.selectbox("Select pipe",df.pipe_id.tolist());b=df[df.pipe_id==pid].iloc[0]
 a,c=st.columns(2)
 with a: age=st.slider("Age",0,100,int(min(100,b.age_years)));soil=st.slider("Soil corrosivity",0,100,int(b.soil_corrosivity));chlor=st.slider("Chloride mg/L",0,500,int(min(500,b.chloride_mg_l)));pressure=st.slider("Pressure bar",0.0,15.0,float(min(15,b.pressure_bar)))
 with c: repairs=st.slider("Repairs / 5y",0,8,int(min(8,b.repair_count_5y)));leaks=st.slider("Leak reports / 3y",0,6,int(min(6,b.leak_reports_3y)));ph=st.slider("pH",4.0,10.0,float(min(10,max(4,b.ph))))
 chem=np.clip(.55*abs(ph-7)/3*100+.45*chlor/400*100,0,100);risk=np.clip(.20*age/80*100+.17*soil+.16*chem+.13*{"PVC":10,"HDPE":8,"Ductile Iron":35,"Cast Iron":65,"Steel":55,"Asbestos Cement":60}.get(b.material,45)+.10*pressure/12*100+.08*repairs/5*100+.07*leaks/4*100+.04*max(0,(b.temperature_c-5)/40*100)+.05*55,0,100)
 m=st.columns(3);m[0].metric("Baseline risk",b.corrosion_score);m[1].metric("Scenario risk",f"{risk:.1f}",f"{risk-b.corrosion_score:+.1f}");m[2].metric("Scenario health",f"{100-risk:.1f}")
 fig=go.Figure(go.Indicator(mode="gauge+number",value=float(risk),title={"text":"Scenario corrosion-risk score"},gauge={"axis":{"range":[0,100]},"bar":{"color":"#0284c7"}}));fig.update_layout(height=310);st.plotly_chart(fig,use_container_width=True)

elif page=="🔔 Rule-Based Alerts":
 alerts=[]
 for _,r in df.iterrows():
  if r.corrosion_score>=75:alerts.append((r.pipe_id,"CRITICAL","Composite risk is critical."))
  if r.age_years>=50:alerts.append((r.pipe_id,"AGE","Advanced asset age."))
  if r.leak_reports_3y>=2:alerts.append((r.pipe_id,"LEAK","Multiple leak reports."))
  if r.repair_count_5y>=3:alerts.append((r.pipe_id,"REPAIR","Recurring repairs."))
  if r.soil_corrosivity>=70:alerts.append((r.pipe_id,"SOIL","Elevated soil corrosivity."))
  if r.chloride_mg_l>=200:alerts.append((r.pipe_id,"CHEMISTRY","Elevated chloride signal."))
  if r.coating_condition=="Poor":alerts.append((r.pipe_id,"COATING","Poor coating condition."))
  if r.cathodic_protection in ["None","Inactive"]:alerts.append((r.pipe_id,"PROTECTION","Limited protection."))
 for pid,typ,msg in alerts: st.markdown(f'<div class="{"alert" if typ in ["CRITICAL","LEAK"] else "warn"}"><b>{pid} · {typ}</b> — {msg}</div>',unsafe_allow_html=True)

elif page=="📂 Data Operations":
 up=st.file_uploader("Upload CSV",type="csv")
 if up:
  raw=pd.read_csv(up);missing=[c for c in REQ if c not in raw.columns]
  if missing:st.error("Missing required columns: "+", ".join(missing));work=df
  else:work=prep(raw);st.success(f"Validated {len(work):,} records locally.")
 else:work=df
 st.dataframe(work,use_container_width=True,hide_index=True);st.download_button("⬇ Download screened CSV",work.to_csv(index=False).encode(),"pipeline_corrosion_screened.csv","text/csv",use_container_width=True)

else:
 st.markdown('<div class="card"><div class="title">⚙️ Methodology & System</div><div class="muted">Transparent local architecture.</div></div>',unsafe_allow_html=True)
 st.markdown("""### Architecture
**Streamlit** interface • **Pandas** CSV processing • **NumPy** deterministic scoring • **Plotly** analytics • **No external APIs**

### Responsible use
This is a corrosion/leakage **screening and decision-support tool**, not a corrosion diagnosis, remaining-life calculation, engineering certification, leak guarantee, or substitute for qualified engineers, field inspection, testing, utility procedures, applicable standards, or emergency response.

All included records are synthetic.
""")
st.markdown('<div class="footer">PipeGuard • Water-Pipeline Corrosion Risk Predictor • Local-first • Synthetic sample data</div>',unsafe_allow_html=True)
