library(lme4)
library(MuMIn)
library(car)


print('Reading Data')
dat = read.csv('./subgraph_task_performance.csv')

#for (subg_id in unique(dat$Subgraph_ID)) {
for (subg_id in c(1, 33, 37, 41, 44, 47, 52)) {
    print(paste('*****************   ', subg_id, '  *****************'))

    dat_sel = dat[(dat$Subgraph_ID==subg_id) & (dat$Task=='Navon') & (dat$Epoch=='Hi'), ]

    model.null = lmer(Accuracy ~ 1 + (1 | Subject_ID), data=dat_sel, REML=FALSE)
    model_pos.real = lmer(Accuracy ~ Expr_Pos + (1 | Subject_ID), data=dat_sel, REML=FALSE)
    model_neg.real = lmer(Accuracy ~ Expr_Neg + (1 | Subject_ID), data=dat_sel, REML=FALSE)

    print(anova(model.null, model_pos.real))
    print(summary(model_pos.real))
    print(' ')

    print(anova(model.null, model_neg.real))
    print(summary(model_neg.real))
    print(' ')
    print(' ')

}
print(warnings())
