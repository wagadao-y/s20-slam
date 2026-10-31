/* Application glue, Apache-2.0. RTKLIB sources retain their BSD license.
 * No file paths or external programs: Rust supplies validated binary frames.
 * postpos has process-global state; the Rust API serialises access. */
#define outsol s20_emit_solution
#include "rtklib.h"
#include "../vendor/rtklib/src/postpos.c"
#undef outsol

typedef int (*progress_fn)(void *, int, double);
typedef void (*solution_fn)(void *, const double *);
static progress_fn progress_cb;
static solution_fn solution_cb;
static void *context;
static int pass;
static int fixed_stats[4];
static double interval;
static unsigned char slip[MAXSAT][NFREQ+NEXOBS];
static unsigned char rover_l2[MAXSAT]; /* bit 1: GPS 2S, bit 2: GPS 2W */
static raw_t *rover;
static rtcm_t *base;

static void copy_signal(obsd_t *o,int dst,int src) {
    o->P[dst]=o->P[src];o->L[dst]=o->L[src];o->D[dst]=o->D[src];
    o->SNR[dst]=o->SNR[src];o->LLI[dst]=o->LLI[src];o->code[dst]=o->code[src];
    o->Pstd[dst]=o->Pstd[src];o->Lstd[dst]=o->Lstd[src];
}
static void clear_signal(obsd_t *o,int idx) {
    o->P[idx]=o->L[idx]=o->D[idx]=o->SNR[idx]=0;
    o->LLI[idx]=o->code[idx]=0;o->Pstd[idx]=o->Lstd[idx]=0;
}
/* Choose a consistent base L2 tracking family for each GPS satellite, after
   partial MSM fragments have been merged. Preserve 2W for legacy/mixed rovers.
   If 2X exists for a 2S-only rover, do not switch back to 2W at missing epochs:
   a tracking-family change can otherwise look like an unflagged phase jump. */
static void match_base_l2(void) {
    unsigned char base_l2x[MAXSAT]={0};
    int i,j;
    for(i=0;i<obss.n;i++) {
        obsd_t *o=&obss.data[i];
        if(o->rcv!=2||satsys(o->sat,NULL)!=SYS_GPS) continue;
        for(j=0;j<NFREQ+NEXOBS;j++)
            if(o->code[j]==CODE_L2X&&o->L[j]!=0) base_l2x[o->sat-1]=1;
    }
    for(i=0;i<obss.n;i++) {
        obsd_t *o=&obss.data[i];
        if(o->rcv!=2||satsys(o->sat,NULL)!=SYS_GPS||
            rover_l2[o->sat-1]!=1||!base_l2x[o->sat-1]) continue;
        for(j=0;j<NFREQ+NEXOBS;j++) if(o->code[j]==CODE_L2X) break;
        if(j<NFREQ+NEXOBS) copy_signal(o,1,j);
        else clear_signal(o,1);
    }
}

extern int showmsg(const char *format, ...) {
    return progress_cb ? progress_cb(context, pass, -1.0) : 0;
}
extern void settspan(gtime_t ts, gtime_t te) {}
extern void settime(gtime_t t) {
    gtime_t u=gpst2utc(t);
    if (progress_cb) progress_cb(context, pass, (double)u.time+u.sec);
}
extern void s20_emit_solution(FILE *fp, const sol_t *s, const double *rb, const solopt_t *o) {
    double p[3], q[9], e[9], v[12];
    gtime_t u=gpst2utc(s->time);
    ecef2pos(s->rr,p);
    q[0]=s->qr[0];q[4]=s->qr[1];q[8]=s->qr[2];
    q[1]=q[3]=s->qr[3];q[5]=q[7]=s->qr[4];q[2]=q[6]=s->qr[5];
    covenu(p,q,e);
    v[0]=(double)u.time+u.sec;v[1]=p[0]*R2D;v[2]=p[1]*R2D;v[3]=p[2];
    v[4]=sqrt(fmax(0,e[0]));v[5]=sqrt(fmax(0,e[4]));v[6]=sqrt(fmax(0,e[8]));
    v[7]=s->ratio;v[8]=s->age;v[9]=s->stat;v[10]=s->ns;
    v[11]=pass;
    if (solution_cb) solution_cb(context,v);
}
static int append_obs(obs_t *src,int rcv) {
    int i,j,keep=1;
    if (!src->n) return 1;
    if (rcv==1) {
        for(i=0;i<src->n;i++) {
            obsd_t *o=&src->data[i];
            int sys=satsys(o->sat,NULL);
            /* Unicore defaults to GPS 2W and puts 2S in an extended slot.
               Modern GPS satellites have 2S only: use it when 2W is absent. */
            if(sys==SYS_GPS && !o->code[1] && o->code[NFREQ+1]==CODE_L2S)
                copy_signal(o,1,NFREQ+1);
            /* Unicore keeps GLONASS 2C outside the three processed bands.
               Use it when the primary L2 signal is absent; AR stays disabled. */
            if(sys==SYS_GLO && !o->code[1] && o->code[NFREQ]==CODE_L2C)
                copy_signal(o,1,NFREQ);
            if(sys==SYS_GPS && o->L[1]!=0) {
                if(o->code[1]==CODE_L2S) rover_l2[o->sat-1]|=1;
                if(o->code[1]==CODE_L2W) rover_l2[o->sat-1]|=2;
            }
        }
        double tow=time2gpst(src->data[0].time,NULL);
        keep=fabs(tow/interval-floor(tow/interval+0.5))*interval<0.001;
        for (i=0;i<src->n;i++) for (j=0;j<NFREQ+NEXOBS;j++)
            slip[src->data[i].sat-1][j]|=src->data[i].LLI[j];
    }
    if (!keep) return 1;
    if (obss.n+src->n>obss.nmax) {
        int n=obss.nmax+65536;
        obsd_t *p=(obsd_t *)realloc(obss.data,n*sizeof(obsd_t));
        if (!p) return 0;
        obss.data=p;obss.nmax=n;
    }
    for (i=0;i<src->n;i++) {
        obsd_t o=src->data[i];o.rcv=rcv;
        if (rcv==2 && satsys(o.sat,NULL)==SYS_CMP) {
            obsd_t in=o;
            for(j=0;j<NFREQ+NEXOBS;j++) {
                o.P[j]=o.L[j]=o.D[j]=o.SNR[j]=0;o.LLI[j]=o.code[j]=0;
            }
            for(j=0;j<NFREQ+NEXOBS;j++) {
                int dst=in.code[j]==CODE_L2I?0:in.code[j]==CODE_L7I?1:in.code[j]==CODE_L6I?2:-1;
                if(dst<0) continue;
                o.P[dst]=in.P[j];o.L[dst]=in.L[j];o.D[dst]=in.D[j];o.SNR[dst]=in.SNR[j];
                o.LLI[dst]=in.LLI[j];o.code[dst]=in.code[j];
            }
        }
        if (rcv==2) {
            int k,sys=satsys(o.sat,NULL);
            /* Some base receivers end EACH MSM fragment, including two BeiDou
               messages at one epoch. sortobs() drops duplicate satellite rows;
               merge their signals here before that can discard the B1I data. */
            for(k=obss.n-1;k>=0;k--) {
                obsd_t *old=&obss.data[k];
                if(old->rcv!=2||fabs(timediff(old->time,o.time))>0.0001) break;
                if(old->sat!=o.sat) continue;
                for(j=0;j<NFREQ+NEXOBS;j++) if(o.code[j]) {
                    int dst=j;
                    if(j<NFREQ && old->code[j] &&
                        getcodepri(sys,old->code[j],"")>getcodepri(sys,o.code[j],"")) continue;
                    old->P[dst]=o.P[j];old->L[dst]=o.L[j];old->D[dst]=o.D[j];
                    old->SNR[dst]=o.SNR[j];old->LLI[dst]|=o.LLI[j];old->code[dst]=o.code[j];
                }
                break;
            }
            if(k>=0 && obss.data[k].rcv==2 && obss.data[k].sat==o.sat &&
                fabs(timediff(obss.data[k].time,o.time))<0.0001) continue;
        }
        if (rcv==1) for (j=0;j<NFREQ+NEXOBS;j++) {
            o.LLI[j]|=slip[o.sat-1][j];slip[o.sat-1][j]=0;
        }
        obss.data[obss.n++]=o;
    }
    return 1;
}
static int append_eph(nav_t *n,int sat,int set) {
    int prn,sys=satsys(sat,&prn);
    if (sys==SYS_GLO) {
        geph_t *p=(geph_t *)realloc(navs.geph,(navs.ng+1)*sizeof(geph_t));
        if (!p) return 0;
        navs.geph=p;navs.geph[navs.ng++]=n->geph[prn-1];navs.ngmax=navs.ng;
    } else if (sat>0) {
        eph_t *p=(eph_t *)realloc(navs.eph,(navs.n+1)*sizeof(eph_t));
        if (!p) return 0;
        navs.eph=p;navs.eph[navs.n++]=n->eph[sat-1+MAXSAT*set];navs.nmax=navs.n;
    }
    return 1;
}
extern int s20_ppk_begin(int week,double tow,double dt,void *ctx,progress_fn p,solution_fn s) {
    memset(&obss,0,sizeof(obss));memset(&navs,0,sizeof(navs));
    memset(&sbss,0,sizeof(sbss));memset(slip,0,sizeof(slip));
    memset(rover_l2,0,sizeof(rover_l2));
    context=ctx;progress_cb=p;solution_cb=s;interval=dt;pass=0;
    rover=(raw_t *)calloc(1,sizeof(raw_t));base=(rtcm_t *)calloc(1,sizeof(rtcm_t));
    if (!rover||!base) return 0;
    if (!init_raw(rover,STRFMT_UNICORE)||!init_rtcm(base)) return 0;
    rover->time=base->time=gpst2time(week,tow);
    strcpy(rover->opt,"-EPHALL");strcpy(base->opt,"-EPHALL -S20BDS");
    return 1;
}
extern int s20_ppk_frame(const unsigned char *b,int len,int rcv) {
    int i,ret=0;
    for(i=0;i<len;i++) ret=rcv==1?input_unicore(rover,b[i]):input_rtcm3(base,b[i]);
    if (ret==1) return append_obs(rcv==1?&rover->obs:&base->obs,rcv);
    if(rcv==2 && ret==0 && len>6) {
        int type=(b[3]<<4)|(b[4]>>4);
        /* Preserve each partial MSM before the next fragment reuses extended
           signal slots. append_obs merges satellites at the same epoch. */
        if(type>=1071 && type<=1137 && type%10>=1 && type%10<=7) return append_obs(&base->obs,2);
    }
    if (ret==2) return append_eph(rcv==1?&rover->nav:&base->nav,
        rcv==1?rover->ephsat:base->ephsat,rcv==1?rover->ephset:base->ephset);
    return ret<0?-1:1;
}
extern int s20_ppk_solve(double elevation,double ratio,int estimate_ionosphere,int systems,int frequencies,int estimate_troposphere,double phase_snr_error,int prefer_base_l2c,double comb_sd_h,double comb_sd_v,double comb_float_sd_h,double comb_float_sd_v) {
    prcopt_t p=prcopt_default;
    solopt_t s=solopt_default;
    rtk_t *r;
    int i,n=0;
    gtime_t start={0},end={0};
    if (!obss.n||!navs.n||norm(base->sta.pos,3)<1E6) return -1;
    if(prefer_base_l2c) match_base_l2();
    /* Keep only rover epochs within the actual base observation coverage. */
    for(i=0;i<obss.n;i++) if(obss.data[i].rcv==2) {
        if(!start.time||timediff(obss.data[i].time,start)<0) start=obss.data[i].time;
        if(!end.time||timediff(obss.data[i].time,end)>0) end=obss.data[i].time;
    }
    if(!start.time) return -1;
    for(i=0;i<obss.n;i++) if(obss.data[i].rcv==2||
        (timediff(obss.data[i].time,start)>=-DTTOL&&timediff(obss.data[i].time,end)<=DTTOL))
        obss.data[n++]=obss.data[i];
    obss.n=n;nepoch=sortobs(&obss);uniqnav(&navs);
    memcpy(navs.glo_fcn,base->nav.glo_fcn,sizeof(navs.glo_fcn));
    p.mode=PMODE_KINEMA;p.soltype=SOLTYPE_COMBINED;p.nf=frequencies;
    p.navsys=systems;
    p.modear=1;p.glomodear=0;p.bdsmodear=1;p.dynamics=1;
    p.minlock=(int)ceil(2.0/interval);p.elmaskar=25.0*D2R;
    p.maxinno[0]=0.2;p.maxinno[1]=10.0;p.niter=2;
    p.ionoopt=estimate_ionosphere?IONOOPT_EST:IONOOPT_BRDC;
    p.tropopt=estimate_troposphere?TROPOPT_EST:TROPOPT_SAAS;
    p.intpref=1;p.elmin=elevation*D2R;
    p.err[6]=phase_snr_error;
    p.thresar[0]=p.thresar[5]=p.thresar[6]=ratio;
    s20_comb_sd_floor[0]=comb_sd_h;s20_comb_sd_floor[1]=comb_sd_v;
    s20_comb_float_sd_floor[0]=comb_float_sd_h;s20_comb_float_sd_floor[1]=comb_float_sd_v;
    p.refpos=0;memcpy(p.rb,base->sta.pos,3*sizeof(double));
    /* RTCM 1005/1006 already contains the antenna reference point ECEF.
       The 1006 antenna height MUST NOT be added a second time. */
    solf=(sol_t *)calloc(nepoch,sizeof(sol_t));solb=(sol_t *)calloc(nepoch,sizeof(sol_t));
    rbf=(double *)calloc(nepoch*3,sizeof(double));rbb=(double *)calloc(nepoch*3,sizeof(double));
    r=(rtk_t *)calloc(1,sizeof(rtk_t));
    if(!solf||!solb||!rbf||!rbb||!r) {free(r);return -2;}
    iobsu=iobsr=isbs=iitm=nitm=isolf=isolb=reverse=aborts=0;
    rtcm_file[0]=rtcm_path[0]=proc_rov[0]=proc_base[0]=0;
    pass=1;rtkinit(r,&p);procpos(NULL,NULL,&p,&s,r,SOLMODE_COMBINED);rtkfree(r);
    for(i=0;i<isolf;i++) s20_emit_solution(NULL,solf+i,rbf+i*3,&s);
    if(!aborts) {
        pass=2;reverse=1;iobsu=iobsr=obss.n-1;isbs=sbss.n-1;
        rtkinit(r,&p);procpos(NULL,NULL,&p,&s,r,SOLMODE_COMBINED);rtkfree(r);
        for(i=isolb-1;i>=0;i--) s20_emit_solution(NULL,solb+i,rbb+i*3,&s);
        if(!aborts) {
            int j;
            memset(fixed_stats,0,sizeof(fixed_stats));
            for(i=0;i<isolf;i++) fixed_stats[0]+=solf[i].stat==SOLQ_FIX;
            for(j=0;j<isolb;j++) fixed_stats[1]+=solb[j].stat==SOLQ_FIX;
            for(i=0,j=isolb-1;i<isolf&&j>=0;i++,j--) {
                double dt=timediff(solf[i].time,solb[j].time);
                if(dt<-DTTOL) {j++;continue;}if(dt>DTTOL) {i--;continue;}
                if(solf[i].stat==SOLQ_FIX&&solb[j].stat==SOLQ_FIX&&
                    !valcomb(solf+i,solb+j,rbf+i*3,rbb+j*3,&p)) fixed_stats[2]++;
                if(((solf[i].stat==SOLQ_FIX&&solb[j].stat==SOLQ_FLOAT)||
                    (solf[i].stat==SOLQ_FLOAT&&solb[j].stat==SOLQ_FIX))&&
                    !valcomb(solf+i,solb+j,rbf+i*3,rbb+j*3,&p)) fixed_stats[3]++;
            }
            pass=3;combres(NULL,NULL,&p,&s);
        }
    }
    free(r);
    return aborts?0:isolf;
}
extern void s20_ppk_end(void) {
    if(rover) {free_raw(rover);free(rover);}if(base) {free_rtcm(base);free(base);}
    rover=NULL;base=NULL;
    freeobsnav(&obss,&navs);
    free(solf);free(solb);free(rbf);free(rbb);
    solf=solb=NULL;rbf=rbb=NULL;
    progress_cb=NULL;solution_cb=NULL;context=NULL;
}
extern void s20_ppk_stats(int *out) { memcpy(out,fixed_stats,sizeof(fixed_stats)); }

/* Raw data inspection: one stream decoded alone, apart from the PPK state above, reporting the
   satellites of each observation epoch. The Rust API serialises access as for PPK. */
static raw_t *insp_rover;
static rtcm_t *insp_base;
extern int s20_inspect_begin(int rcv,int week,double tow) {
    if (rcv==1) {
        insp_rover=(raw_t *)calloc(1,sizeof(raw_t));
        if (!insp_rover||!init_raw(insp_rover,STRFMT_UNICORE)) return 0;
        insp_rover->time=gpst2time(week,tow);strcpy(insp_rover->opt,"-EPHALL");
    } else {
        insp_base=(rtcm_t *)calloc(1,sizeof(rtcm_t));
        if (!insp_base||!init_rtcm(insp_base)) return 0;
        insp_base->time=gpst2time(week,tow);strcpy(insp_base->opt,"-EPHALL -S20BDS");
    }
    return 1;
}
/* Feeds one frame. When it gives observations (an epoch, or a base MSM fragment), writes their
   UTC time and per satellite (at most max): number, system, best SNR (dB-Hz), signals tracked and
   whether a signal with carrier phase slipped. Returns the satellites written, 0 for other messages, -1 on an error. */
extern int s20_inspect_frame(const unsigned char *b,int len,int rcv,double *utc,int *sat,int *sys,
    double *snr,int *sig,int *slp,int max) {
    int i,j,k,n,ret=0;
    obs_t *o=rcv==1?&insp_rover->obs:&insp_base->obs;
    for(i=0;i<len;i++) ret=rcv==1?input_unicore(insp_rover,b[i]):input_rtcm3(insp_base,b[i]);
    if (ret<0) return -1;
    if (ret!=1) {
        int type=len>6?(b[3]<<4)|(b[4]>>4):0;
        if (!(rcv==2 && ret==0 && type>=1071 && type<=1137 && type%10>=1 && type%10<=7)) return 0;
    }
    if (o->n<=0) return 0;
    gtime_t t=gpst2utc(o->data[0].time);
    *utc=(double)t.time+t.sec;
    n=o->n<max?o->n:max;
    for(k=0;k<n;k++) {
        const obsd_t *d=&o->data[k];
        sat[k]=d->sat;sys[k]=satsys(d->sat,NULL);snr[k]=0;sig[k]=0;slp[k]=0;
        for(j=0;j<NFREQ+NEXOBS;j++) if (d->code[j]) {
            sig[k]++;
            if (d->SNR[j]>snr[k]) snr[k]=d->SNR[j];
            /* a signal without carrier phase (its lock time stays 0) is flagged at every epoch
               by the Unicore decoder: not a slip */
            if ((d->LLI[j]&1) && d->L[j]!=0.0) slp[k]=1;
        }
    }
    return n;
}
extern void s20_inspect_end(void) {
    if(insp_rover) {free_raw(insp_rover);free(insp_rover);}
    if(insp_base) {free_rtcm(insp_base);free(insp_base);}
    insp_rover=NULL;insp_base=NULL;
}
/* Latitude and longitude (deg) and ellipsoidal height of an ECEF position. */
extern void s20_ecef_to_llh(const double *r,double *llh) {
    ecef2pos(r,llh);llh[0]*=R2D;llh[1]*=R2D;
}

/* Regression fixture exercises exactly the application glue used in decoding. */
extern int s20_ppk_test_signals(void) {
    obsd_t g={0},b={0},r={0};obs_t obs={0};int ok=1;
    obs.n=obs.nmax=1;obs.data=&g;
    if(!s20_ppk_begin(2439,263538.,0.1,NULL,NULL,NULL)) {s20_ppk_end();return 0;}
    g.time=gpst2time(2439,263538.2);g.sat=satno(SYS_GPS,1);
    g.code[NFREQ+1]=CODE_L2S;g.L[NFREQ+1]=44.;
    append_obs(&obs,1);
    ok &= obss.n==1&&obss.data[0].code[1]==CODE_L2S&&obss.data[0].L[1]==44.;
    g.time=gpst2time(2439,263538.25);g.code[1]=0;g.LLI[NFREQ+1]=LLI_SLIP;
    append_obs(&obs,1);
    g.time=gpst2time(2439,263538.3);g.code[1]=0;g.LLI[NFREQ+1]=g.LLI[1]=0;
    append_obs(&obs,1);
    ok &= obss.n==2&&(obss.data[1].LLI[1]&LLI_SLIP);
    b.time=gpst2time(2439,263538.2);b.sat=satno(SYS_CMP,10);
    b.code[0]=CODE_L2I;b.L[0]=10.;obs.data=&b;
    append_obs(&obs,2);
    b.code[0]=0;b.L[0]=0;b.code[3]=CODE_L6I;b.L[3]=30.;
    append_obs(&obs,2);
    ok &= obss.n==3&&obss.data[2].code[0]==CODE_L2I&&obss.data[2].L[0]==10.
        &&obss.data[2].code[2]==CODE_L6I&&obss.data[2].L[2]==30.;
    r.time=gpst2time(2439,263538.2);r.sat=satno(SYS_GLO,1);
    r.code[NFREQ]=CODE_L2C;r.L[NFREQ]=24.;r.P[NFREQ]=2E7;
    r.SNR[NFREQ]=140;r.Lstd[NFREQ]=0.01;r.Pstd[NFREQ]=1.5;obs.data=&r;
    append_obs(&obs,1);
    ok &= obss.n==4&&obss.data[3].code[1]==CODE_L2C&&obss.data[3].L[1]==24.
        &&obss.data[3].P[1]==2E7&&obss.data[3].SNR[1]==140
        &&obss.data[3].Lstd[1]==r.Lstd[NFREQ]&&obss.data[3].Pstd[1]==r.Pstd[NFREQ];
    r.time=gpst2time(2439,263538.25);r.code[1]=0;r.LLI[NFREQ]=LLI_SLIP;
    append_obs(&obs,1);
    r.time=gpst2time(2439,263538.3);r.code[1]=0;r.LLI[NFREQ]=r.LLI[1]=0;
    append_obs(&obs,1);
    ok &= obss.n==5&&(obss.data[4].LLI[1]&LLI_SLIP);
    /* An available primary 2P must not be overwritten by extended 2C. */
    r.time=gpst2time(2439,263538.4);r.code[1]=CODE_L2P;r.L[1]=12.;
    append_obs(&obs,1);
    ok &= obss.n==6&&obss.data[5].code[1]==CODE_L2P&&obss.data[5].L[1]==12.;
    s20_ppk_end();return ok;
}

extern int s20_ppk_test_l2_selection(void) {
    obsd_t o={0};obs_t obs={0};int ok=1,i;
    obs.n=obs.nmax=1;obs.data=&o;
    if(!s20_ppk_begin(2439,263538.,0.1,NULL,NULL,NULL)) {s20_ppk_end();return 0;}
    /* Satellite 1 uses 2S, satellite 2 uses 2W, 3 switches, 4 has no base 2X. */
    for(i=1;i<=4;i++) {
        memset(&o,0,sizeof(o));o.time=gpst2time(2439,263538.2);
        o.sat=satno(SYS_GPS,i);o.code[1]=i==2?CODE_L2W:CODE_L2S;o.L[1]=44.;
        append_obs(&obs,1);
    }
    o.sat=satno(SYS_GPS,3);o.time=gpst2time(2439,263538.3);
    o.code[1]=CODE_L2W;append_obs(&obs,1);
    for(i=1;i<=4;i++) {
        memset(&o,0,sizeof(o));o.time=gpst2time(2439,263538.2);
        o.sat=satno(SYS_GPS,i);o.code[1]=CODE_L2W;o.L[1]=11.;
        if(i!=4) {
            o.code[NFREQ]=CODE_L2X;o.L[NFREQ]=22.;o.P[NFREQ]=2E7;
            o.LLI[NFREQ]=LLI_SLIP;o.SNR[NFREQ]=140;
            o.Lstd[NFREQ]=0.01;o.Pstd[NFREQ]=1.5;
        }
        append_obs(&obs,2);
    }
    memset(&o,0,sizeof(o));o.time=gpst2time(2439,263539.2);
    o.sat=satno(SYS_GPS,1);o.code[1]=CODE_L2W;o.L[1]=11.;
    append_obs(&obs,2);
    match_base_l2();
    ok &= obss.n==10&&obss.data[5].code[1]==CODE_L2X&&obss.data[5].L[1]==22.
        &&obss.data[5].P[1]==2E7&&(obss.data[5].LLI[1]&LLI_SLIP)
        &&obss.data[5].SNR[1]==140&&obss.data[5].Pstd[1]==1.5;
    ok &= obss.data[6].code[1]==CODE_L2W&&obss.data[6].L[1]==11.;
    ok &= obss.data[7].code[1]==CODE_L2W&&obss.data[7].L[1]==11.;
    ok &= obss.data[8].code[1]==CODE_L2W&&obss.data[8].L[1]==11.;
    ok &= obss.data[9].code[1]==0&&obss.data[9].L[1]==0;
    s20_ppk_end();return ok;
}

/* Feed real-format binary frames through both receiver decoders, not prebuilt
   observations. Old MAXPRNCMP=46 ignored higher-PRN observations and rejected
   the recorded C56/C58 RTCM 1042 ephemerides. Exercise the C63 boundary as well. */
extern int s20_ppk_test_bds3(void) {
    unsigned char raw[72]={0},rtcm[70]={0};
    uint16_t week=2439,cnr=4500,prn=56;
    uint32_t tow=263538200,crc,track=(4u<<16)|(1u<<10)|(1u<<12);
    double pseudorange=2E7,phase=-100000.;float lock=5.;int ok;
    if(!s20_ppk_begin(week,263538.2,0.1,NULL,NULL,NULL)) {s20_ppk_end();return 0;}
    raw[0]=0xaa;raw[1]=0x44;raw[2]=0xb5;raw[4]=12;raw[6]=44;
    memcpy(raw+10,&week,2);memcpy(raw+12,&tow,4);raw[24]=1;
    memcpy(raw+30,&prn,2);memcpy(raw+32,&pseudorange,8);memcpy(raw+40,&phase,8);
    memcpy(raw+56,&cnr,2);memcpy(raw+60,&lock,4);memcpy(raw+64,&track,4);
    crc=rtk_crc32(raw,68);memcpy(raw+68,&crc,4);
    ok=s20_ppk_frame(raw,sizeof(raw),1)==1&&obss.n==1
        &&obss.data[0].sat==satno(SYS_CMP,56)&&obss.data[0].L[0]==100000.;
    prn=63;tow+=100;track|=21u<<21; /* Unicore B3I */
    memcpy(raw+30,&prn,2);memcpy(raw+12,&tow,4);memcpy(raw+64,&track,4);
    crc=rtk_crc32(raw,68);memcpy(raw+68,&crc,4);
    ok &= s20_ppk_frame(raw,sizeof(raw),1)==1&&obss.n==2
        &&obss.data[1].sat==satno(SYS_CMP,63)&&obss.data[1].L[2]==100000.
        &&satno(SYS_CMP,64)==0;
    rtcm[0]=0xd3;rtcm[2]=64;setbitu(rtcm,24,12,1042);
    setbitu(rtcm,36,6,58);setbitu(rtcm,42,13,1083);
    crc=rtk_crc24q(rtcm,67);rtcm[67]=crc>>16;rtcm[68]=crc>>8;rtcm[69]=crc;
    ok &= s20_ppk_frame(rtcm,sizeof(rtcm),2)==1&&navs.n==1
        &&navs.eph[0].sat==satno(SYS_CMP,58);
    s20_ppk_end();return ok;
}

typedef struct { int status;double up_sd,east_sd; } test_result_t;
static void test_status(void *ctx,const double *v) {
    test_result_t *r=(test_result_t *)ctx;r->status=(int)v[9];r->up_sd=v[6];r->east_sd=v[4];
}
extern int s20_ppk_test_mixed_consistency(void) {
    prcopt_t p=prcopt_default;solopt_t s=solopt_default;test_result_t actual={0};int ok=1,k;
    if(!s20_ppk_begin(2439,263538.2,0.1,&actual,NULL,test_status)) {s20_ppk_end();return 0;}
    solf=(sol_t *)calloc(1,sizeof(sol_t));solb=(sol_t *)calloc(1,sizeof(sol_t));
    rbf=(double *)calloc(3,sizeof(double));rbb=(double *)calloc(3,sizeof(double));
    if(!solf||!solb||!rbf||!rbb) {s20_ppk_end();return 0;}
    isolf=isolb=1;iitm=nitm=0;pass=3;p.mode=PMODE_KINEMA;p.dynamics=0;
    s20_comb_sd_floor[0]=s20_comb_sd_floor[1]=0;
    s20_comb_float_sd_floor[0]=s20_comb_float_sd_floor[1]=0;
    solf[0].time=solb[0].time=gpst2time(2439,263538.2);
    solf[0].rr[0]=6378137.;
    for(k=0;k<3;k++) solf[0].qr[k]=solb[0].qr[k]=0.0001f;
    solf[0].stat=SOLQ_FIX;solb[0].stat=SOLQ_FLOAT;solb[0].rr[0]=6378137.01;
    combres(NULL,NULL,&p,&s);ok &= actual.status==SOLQ_FIX&&actual.up_sd<0.011;
    actual.status=0;solb[0].rr[0]=6378138.;
    combres(NULL,NULL,&p,&s);ok &= actual.status==SOLQ_FIX&&actual.up_sd>=1.;
    actual.status=0;solf[0].stat=SOLQ_FLOAT;solb[0].stat=SOLQ_FIX;
    combres(NULL,NULL,&p,&s);ok &= actual.status==SOLQ_FIX&&actual.up_sd>=1.;
    actual.status=0;solf[0].stat=SOLQ_FIX;
    combres(NULL,NULL,&p,&s);ok &= actual.status==SOLQ_FLOAT; /* original FIX/FIX gate */
    actual.status=0;solb[0].stat=SOLQ_SINGLE;
    combres(NULL,NULL,&p,&s);ok &= actual.status==SOLQ_FIX&&actual.up_sd<0.011;
    /* At longitude 45 degrees, (1,-1,0) is horizontal. Omitting/reordering
       covariance cross terms would incorrectly increase vertical uncertainty. */
    solb[0].stat=SOLQ_FLOAT;actual.status=0;
    solf[0].rr[0]=solf[0].rr[1]=6378137./sqrt(2.);
    solb[0].rr[0]=solf[0].rr[0]+1.;solb[0].rr[1]=solf[0].rr[1]-1.;
    combres(NULL,NULL,&p,&s);
    ok &= actual.status==SOLQ_FIX&&actual.east_sd>1.41&&actual.up_sd<0.011;
    s20_ppk_end();return ok;
}

/* FIX/FIX directional check in ENU with sigma floors. At latitude/longitude 0,
   ECEF x is up, y east and z north. Millimetre FIX covariance alone rejects a
   2 cm difference; the floors accept centimetres but not decimetres. */
extern int s20_ppk_test_fix_floor(void) {
    prcopt_t p=prcopt_default;solopt_t s=solopt_default;test_result_t actual={0};int ok=1,k;
    static const struct { double d[3],fh,fv;int stat; } c[]={
        {{0,0.02,0},0,0,SOLQ_FLOAT},       /* filter sigma only */
        {{0,0.02,0},0.01,0.02,SOLQ_FIX},   /* 2 cm east */
        {{0,0.03,0.03},0.01,0.02,SOLQ_FIX},/* 3 cm east and north */
        {{0.06,0,0},0.01,0.02,SOLQ_FIX},   /* 6 cm up */
        {{0,0.06,0},0.01,0.02,SOLQ_FLOAT}, /* 6 cm east */
        {{0.10,0,0},0.01,0.02,SOLQ_FLOAT}, /* 10 cm up */
        {{0,0,0.19},0.01,0.02,SOLQ_FLOAT}, /* one L1 wavelength north */
    };
    if(!s20_ppk_begin(2439,263538.2,0.1,&actual,NULL,test_status)) {s20_ppk_end();return 0;}
    solf=(sol_t *)calloc(1,sizeof(sol_t));solb=(sol_t *)calloc(1,sizeof(sol_t));
    rbf=(double *)calloc(3,sizeof(double));rbb=(double *)calloc(3,sizeof(double));
    if(!solf||!solb||!rbf||!rbb) {s20_ppk_end();return 0;}
    isolf=isolb=1;iitm=nitm=0;pass=3;p.mode=PMODE_KINEMA;p.dynamics=0;
    solf[0].time=solb[0].time=gpst2time(2439,263538.2);
    solf[0].stat=solb[0].stat=SOLQ_FIX;
    for(k=0;k<3;k++) solf[0].qr[k]=solb[0].qr[k]=1E-6f; /* 1 mm */
    for(size_t i=0;i<sizeof(c)/sizeof(c[0]);i++) {
        s20_comb_sd_floor[0]=c[i].fh;s20_comb_sd_floor[1]=c[i].fv;
        solf[0].rr[0]=6378137.;solf[0].rr[1]=solf[0].rr[2]=0;
        for(k=0;k<3;k++) solb[0].rr[k]=solf[0].rr[k]+c[i].d[k];
        actual.status=0;combres(NULL,NULL,&p,&s);
        ok &= actual.status==c[i].stat;
    }
    s20_comb_sd_floor[0]=s20_comb_sd_floor[1]=0;
    s20_comb_float_sd_floor[0]=s20_comb_float_sd_floor[1]=0;
    s20_ppk_end();return ok;
}

/* FIX/FLOAT check with the FLOAT floors (axes as above): a converging FLOAT
   decimetres away keeps the FIX sigma, metres away still enlarge it, and
   FIX/FIX pairs ignore the FLOAT floors. */
extern int s20_ppk_test_float_floor(void) {
    prcopt_t p=prcopt_default;solopt_t s=solopt_default;test_result_t actual={0};int ok=1,k;
    static const struct { double d[3],g;int other,stat;double east_sd,up_sd; } c[]={
        {{0,0.30,0},0.10,SOLQ_FLOAT,SOLQ_FIX,0.011,0.011},  /* 30 cm east */
        {{0,0.30,0},0,SOLQ_FLOAT,SOLQ_FIX,1E9,1E9},         /* no FLOAT floors: enlarged */
        {{0.70,0,0},0.10,SOLQ_FLOAT,SOLQ_FIX,0.011,0.011},  /* 70 cm up */
        {{0,0.50,0},0.10,SOLQ_FLOAT,SOLQ_FIX,1E9,0.011},    /* 50 cm east: enlarged */
        {{0.90,0,0},0.10,SOLQ_FLOAT,SOLQ_FIX,0.011,1E9},    /* 90 cm up: enlarged */
        {{0,0.30,0},0.10,SOLQ_FIX,SOLQ_FLOAT,0,0},          /* FIX/FIX: downgraded */
    };
    if(!s20_ppk_begin(2439,263538.2,0.1,&actual,NULL,test_status)) {s20_ppk_end();return 0;}
    solf=(sol_t *)calloc(1,sizeof(sol_t));solb=(sol_t *)calloc(1,sizeof(sol_t));
    rbf=(double *)calloc(3,sizeof(double));rbb=(double *)calloc(3,sizeof(double));
    if(!solf||!solb||!rbf||!rbb) {s20_ppk_end();return 0;}
    isolf=isolb=1;iitm=nitm=0;pass=3;p.mode=PMODE_KINEMA;p.dynamics=0;
    solf[0].time=solb[0].time=gpst2time(2439,263538.2);
    for(k=0;k<3;k++) solf[0].qr[k]=solb[0].qr[k]=1E-6f; /* 1 mm */
    s20_comb_sd_floor[0]=0.01;s20_comb_sd_floor[1]=0.02;
    for(size_t i=0;i<sizeof(c)/sizeof(c[0]);i++) {
        s20_comb_float_sd_floor[0]=c[i].g;s20_comb_float_sd_floor[1]=2*c[i].g;
        solf[0].stat=SOLQ_FIX;solb[0].stat=c[i].other;
        solf[0].rr[0]=6378137.;solf[0].rr[1]=solf[0].rr[2]=0;
        for(k=0;k<3;k++) solb[0].rr[k]=solf[0].rr[k]+c[i].d[k];
        actual.status=0;combres(NULL,NULL,&p,&s);
        ok &= actual.status==c[i].stat;
        if(c[i].stat==SOLQ_FIX) {
            /* 1E9: enlarged by the separation, else at most the FIX sigma */
            ok &= c[i].east_sd>1.?actual.east_sd>=c[i].d[1]:actual.east_sd<c[i].east_sd;
            ok &= c[i].up_sd>1.?actual.up_sd>=c[i].d[0]:actual.up_sd<c[i].up_sd;
        }
    }
    s20_comb_sd_floor[0]=s20_comb_sd_floor[1]=0;
    s20_comb_float_sd_floor[0]=s20_comb_float_sd_floor[1]=0;
    s20_ppk_end();return ok;
}
