suppressPackageStartupMessages(library(survival))
arg <- grep('^--file=',commandArgs(),value=TRUE)
an <- dirname(normalizePath(sub('^--file=','',arg[1])))
out <- file.path(an,'enhancement_results')
d <- read.csv(file.path(out,'survival_inputs.csv'))
stopifnot(nrow(d)==45, sum(d$event)==41,all(d$time>0))
coefrows <- list(); phrows <- list(); i<-0;j<-0
for (s in 1:4) {
 d$indicator <- as.integer(d$subtype==s)
 for (kind in c('unadjusted','adjusted','score','score_purity')) {
  d$score <- d[[paste0('score',s)]]
  f <- switch(kind,unadjusted=Surv(time,event)~indicator,
     adjusted=Surv(time,event)~indicator+purity+brain,
     score=Surv(time,event)~score,score_purity=Surv(time,event)~score+purity)
  fit <- coxph(f,data=d,ties='breslow',x=TRUE,y=TRUE)
  a <- summary(fit); ci <- a$conf.int
  i<-i+1;coefrows[[i]]<-data.frame(subtype=s,model=kind,term=rownames(ci),HR=ci[,1],lo=ci[,3],hi=ci[,4],p=a$coefficients[,5])
  z <- cox.zph(fit,transform='km')$table
  j<-j+1;phrows[[j]]<-data.frame(subtype=s,model=kind,term=rownames(z),chisq=z[,1],df=z[,2],p=z[,3])
 }
}
write.csv(do.call(rbind,coefrows),file.path(out,'survival_cox_r.csv'),row.names=FALSE)
ph<-do.call(rbind,phrows); ph$q_global_16<-NA_real_
global<-ph$term=='GLOBAL'; ph$q_global_16[global]<-p.adjust(ph$p[global],'BH')
write.csv(ph,file.path(out,'survival_ph_diagnostics.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,'R_sessionInfo.txt'))
print(ph)
