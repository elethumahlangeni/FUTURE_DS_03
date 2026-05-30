import pandas as pd, numpy as np, json, os

PATH = "/sessions/lucid-amazing-rubin/mnt/Data Science/Task 3/Bank Marketing/bank-additional/bank-additional/bank-additional-full.csv"
df = pd.read_csv(PATH, sep=';')

# ── MAPPINGS ─────────────────────────────────────────────────────────────────
JOB    = sorted(df['job'].unique().tolist())
MAR    = sorted(df['marital'].unique().tolist())
EDU    = ['basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown']
MONTHS = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
DAYS   = ['mon','tue','wed','thu','fri']
POUT   = ['failure','nonexistent','success']
CON    = ['cellular','telephone']
HOU    = ['no','unknown','yes']
LOA    = ['no','unknown','yes']
AGE_L  = ['18-25','26-35','36-45','46-55','56-65','65+']
DUR_L  = ['No Answer','<1 min','1-2 min','2-5 min','5-10 min','10+ min']
CAM_L  = ['1 Call','2 Calls','3 Calls','4-5','6-10','11+']

jm={v:i for i,v in enumerate(JOB)}; mm={v:i for i,v in enumerate(MAR)}
em={v:i for i,v in enumerate(EDU)}; monm={v:i for i,v in enumerate(MONTHS)}
dm={v:i for i,v in enumerate(DAYS)}; pm={v:i for i,v in enumerate(POUT)}
cm={v:i for i,v in enumerate(CON)}; hm2={v:i for i,v in enumerate(HOU)}
lm={v:i for i,v in enumerate(LOA)}

def ag(a):
    if a<=25:return 0
    if a<=35:return 1
    if a<=45:return 2
    if a<=55:return 3
    if a<=65:return 4
    return 5

def db(d):
    if d==0:return 0
    if d<60:return 1
    if d<120:return 2
    if d<300:return 3
    if d<600:return 4
    return 5

def cb(c):
    if c==1:return 0
    if c==2:return 1
    if c==3:return 2
    if c<=5:return 3
    if c<=10:return 4
    return 5

# ── COMPACT RAW STRING (14 chars/row) ─────────────────────────────────────────
rows=[]
for _,r in df.iterrows():
    rows.append(f"{ag(r['age'])}{jm.get(r['job'],0):02d}{mm.get(r['marital'],0)}{em.get(r['education'],7)}{cm.get(r['contact'],0)}{monm.get(r['month'],0):02d}{pm.get(r['poutcome'],1)}{hm2.get(r['housing'],1)}{lm.get(r['loan'],1)}{db(r['duration'])}{cb(r['campaign'])}{1 if r['y']=='yes' else 0}")
raw_str=''.join(rows)

# ── AGGREGATIONS ──────────────────────────────────────────────────────────────
def agg(s):
    n=len(s); c=(s['y']=='yes').sum()
    return {'n':int(n),'c':int(c),'r':round(c/n*100,2) if n else 0}

df['_ag']=df['age'].apply(ag)
df['_db']=df['duration'].apply(db)
df['_cb']=df['campaign'].apply(cb)
df['_yb']=(df['y']=='yes').astype(int)

total=len(df); cont=int((df['duration']>0).sum()); eng=int((df['duration']>120).sum())
qual=int(((df['duration']>300)|(df['poutcome']=='success')).sum()); conv_n=int((df['y']=='yes').sum())

by_month={m:agg(df[df['month']==m]) for m in MONTHS if m in df['month'].values}
by_job={j:agg(df[df['job']==j]) for j in JOB}
by_age={AGE_L[i]:agg(df[df['_ag']==i]) for i in range(6)}
by_edu={e:agg(df[df['education']==e]) for e in EDU if e in df['education'].values}
by_mar={m:agg(df[df['marital']==m]) for m in MAR}
by_con={c:{**agg(df[df['contact']==c]),'eng':int(((df['contact']==c)&(df['duration']>120)).sum()),'qual':int(((df['contact']==c)&((df['duration']>300)|(df['poutcome']=='success'))).sum())} for c in CON}
by_pout={p:agg(df[df['poutcome']==p]) for p in POUT}
by_dur={DUR_L[i]:agg(df[df['_db']==i]) for i in range(6)}
by_cam={CAM_L[i]:agg(df[df['_cb']==i]) for i in range(6)}
by_hou={h:agg(df[df['housing']==h]) for h in HOU if h in df['housing'].values}
by_loa={l:agg(df[df['loan']==l]) for l in LOA if l in df['loan'].values}
by_day={d:agg(df[df['day_of_week']==d]) for d in DAYS}

# Heatmap job x month
hm={}
for j in JOB:
    hm[j]={}
    for mo in MONTHS:
        s=df[(df['job']==j)&(df['month']==mo)]
        hm[j][mo]=round(float((s['y']=='yes').mean()*100),1) if len(s)>10 else None

# Feature importance (point-biserial correlation ×100)
fi={}
fi['Call Duration']     =round(float(df['duration'].corr(df['_yb']))*100,2)
fi['Previous Success']  =round(float((df['poutcome']=='success').astype(int).corr(df['_yb']))*100,2)
fi['Prior Contact']     =round(float((df['previous']>0).astype(int).corr(df['_yb']))*100,2)
fi['Age 65+']           =round(float((df['age']>=65).astype(int).corr(df['_yb']))*100,2)
fi['Cellular Channel']  =round(float((df['contact']=='cellular').astype(int).corr(df['_yb']))*100,2)
fi['Student/Retired']   =round(float(df['job'].isin(['student','retired']).astype(int).corr(df['_yb']))*100,2)
fi['University Degree'] =round(float((df['education']=='university.degree').astype(int).corr(df['_yb']))*100,2)
fi['No Personal Loan']  =round(float((df['loan']=='no').astype(int).corr(df['_yb']))*100,2)
fi['High Call Freq']    =round(float((df['campaign']>5).astype(int).corr(df['_yb']))*100,2)
fi['Blue-Collar']       =round(float((df['job']=='blue-collar').astype(int).corr(df['_yb']))*100,2)
fi['Has Housing Loan']  =round(float((df['housing']=='yes').astype(int).corr(df['_yb']))*100,2)

# Best month (>=500 contacts)
mo_df=pd.DataFrame([{'m':k,'r':v['r'],'n':v['n']} for k,v in by_month.items()])
best_mo=mo_df[mo_df['n']>=500].sort_values('r',ascending=False).iloc[0]

payload=dict(raw=raw_str,
  maps=dict(job=JOB,mar=MAR,edu=EDU,month=MONTHS,day=DAYS,poutcome=POUT,contact=CON,housing=HOU,loan=LOA,age=AGE_L,dur=DUR_L,cam=CAM_L),
  kpi=dict(total=total,contacted=cont,engaged=eng,qualified=qual,converted=conv_n,
    conv_rate=round(conv_n/total*100,2),avg_dur=round(float(df['duration'].mean()),1),
    avg_cam=round(float(df['campaign'].mean()),2),
    best_month=str(best_mo['m']).capitalize(),best_month_rate=round(float(best_mo['r']),1)),
  funnel=dict(stages=['Contacted','Reached','Interested','Repeat Contact','Converted'],
    volumes=[total,cont,eng,qual,conv_n]),
  by_month=by_month,by_job=by_job,by_age=by_age,by_edu=by_edu,by_mar=by_mar,
  by_con=by_con,by_pout=by_pout,by_dur=by_dur,by_cam=by_cam,
  by_hou=by_hou,by_loa=by_loa,by_day=by_day,heatmap=hm,fi=fi)

with open('/tmp/payload.json','w') as f:
    json.dump(payload,f,separators=(',',':'))
print("Payload:",round(os.path.getsize('/tmp/payload.json')/1024),"KB")
print("KPIs:",payload['kpi'])
