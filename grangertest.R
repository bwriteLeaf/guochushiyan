table <- read.csv("C:/Users/user/Documents/YJS/国储局Ⅱ期/实验/guochushiyan/senti-res-1.csv", sep=",", header=T)
senti = table[["sentiment"]]
wti = table[["wti"]]
library(lmtest)  
grangertest(senti~wti, order = 2, data =table)  
grangertest(wti~senti, order = 2, data =table)  