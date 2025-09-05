# Baseline table and propensity score matching
# install.packages("tableone")
library(tableone)
raw.data <- read.csv(file = "data_baseline_english.csv", fileEncoding = "gbk")

# Method: Use the within function to group the BMI variable
raw.data <- within(raw.data,{
  BMI_Type <- NA
  BMI_Type[raw.data$BMI<18.5]="U"
  BMI_Type[raw.data$BMI>= 18.5 & raw.data$BMI < 23.0] = "N"
  BMI_Type[raw.data$BMI>= 23.0 & raw.data$BMI < 27.5] = "O"
  BMI_Type[raw.data$BMI>= 27.5] = "OB"
})
# print(raw.data$group)
dput(names(raw.data))

raw.data$label <- 1-raw.data$label
raw.data <- raw.data[c("X", "patient_sn", 
                       "Body.temperature", "BMI_Type", "Respiratory.rate", 
                       "age", "smoking", "Operation.history", "diabetes", 
                       "Sex", "drink", "Hy", 
                       "label")]


myVars <- c("Body.temperature", "BMI_Type", "Respiratory.rate",
            "age", "smoking", "Operation.history", "diabetes", "Sex",
            "drink", "Hy")  # 10

catVars <- c("smoking", "Operation.history", "diabetes", "Sex", "drink", "Hy",  "BMI_Type")

tab1 <- CreateTableOne(vars = myVars, strata = "label" ,data = raw.data, factorVars = catVars)
tab1 <- print(tab1,  exact = "extent", smd = TRUE)
#################################################################################################
# Body.temperature + Length.of.stay + Operation.duration + SBP + BMI_Type + 
# Respiratory.rate + Heart.rate + age + smoking + Operation.history + diabetes + Sex +
# Cerebral.infarction + drink + Hy + Lower.limb.edema
library(MatchIt)
matchlist <- matchit(label ~ Body.temperature + Respiratory.rate + age + smoking + diabetes + 
                       Sex +  Operation.history + drink + Hy + BMI_Type,
                     data = raw.data,
                     method   = "nearest",
                     distance = "logit",
                     caliper  = 0.02,
                     ratio    = 1,
                     replace  = F,
                     discard = "control")

matchdata <- match.data(matchlist)
nrow(matchdata)
dput(names(matchdata))
# head(matchdata)
write.csv(matchdata,file = "data_result_matched.csv")


#########################################################################################

tab2 <- CreateTableOne(vars = myVars, strata = "label" ,data = matchdata, factorVars = catVars)
tab2 <- print(tab2,  exact = "extent", smd = TRUE)

# Merge and output the matched quantitative results
merged_df <- cbind(tab1, tab2)
write.csv(merged_df,file = "result_matched.csv")
write.csv(tab1,file = "result_before.csv")
write.csv(tab2,file = "result_after.csv")


