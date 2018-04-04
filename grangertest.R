#isweek = TRUE
isweek = FALSE

infile = "C:/Users/user/Documents/YJS/国储局Ⅱ期/实验/guochushiyan/winequality/wti-senti-hp.csv"
if (isweek){
  infile = "C:/Users/user/Documents/YJS/国储局Ⅱ期/实验/guochushiyan/winequality/wti-senti-week-hp.csv"
}

table <- read.csv(infile, sep=",", header=T)
senti = table[["sentiment"]]
wti = table[["wti"]]
library(lmtest)  
grangertest(senti~wti, order = 20, data =table)  
#grangertest(wti~senti, order = 1, data =table)  
