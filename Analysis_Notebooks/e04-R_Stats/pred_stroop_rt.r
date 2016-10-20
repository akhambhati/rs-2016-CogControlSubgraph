library(lme4)
library(MuMIn)
library(car)


print('Reading Data')
dat = read.csv('./subgraph_task_performance.csv')

#for (subg_id in unique(dat$Subgraph_ID)) {
for (subg_id in c(7, 14, 19, 49)) {
    print(paste('*****************   ', subg_id, '  *****************'))

    dat_sel = dat[(dat$Subgraph_ID==subg_id) & (dat$Task=='Stroop') & (dat$Epoch=='Hi'), ]

    model.null = lmer(Median_RT ~ 1 + (1 | Subject_ID), data=dat_sel, REML=FALSE)
    model_pos.real = lmer(Median_RT ~ Expr_Pos + (1 | Subject_ID), data=dat_sel, REML=FALSE)
    model_neg.real = lmer(Median_RT ~ Expr_Neg + (1 | Subject_ID), data=dat_sel, REML=FALSE)

    print(anova(model.null, model_pos.real))
    print(summary(model_pos.real))
    print(' ')

    print(anova(model.null, model_neg.real))
    print(summary(model_neg.real))
    print(' ')
    print(' ')

}
print(warnings())
