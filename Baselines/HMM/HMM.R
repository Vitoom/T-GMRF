library(depmixS4)

data(speed)
mod <- depmix(list(rt~1,corr~1),data=speed,transition=~Pacc,nstates=2,
family=list(gaussian(),multinomial("identity")),ntimes=c(168,134,137))
fmod <- fit(mod)

pst_global <- posterior(fmod, type = "global")
