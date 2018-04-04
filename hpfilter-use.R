#两个一起做趋势化，用于grangertest
#isweek = TRUE
isweek = FALSE

infile = "C:/Users/user/Documents/YJS/国储局Ⅱ期/实验/guochushiyan/winequality/wti-senti.csv"
outfile = "C:/Users/user/Documents/YJS/国储局Ⅱ期/实验/guochushiyan/winequality/wti-senti-hp-g.csv"
if (isweek){
  infile = "C:/Users/user/Documents/YJS/国储局Ⅱ期/实验/guochushiyan/winequality/wti-senti-week.csv"
  outfile = "C:/Users/user/Documents/YJS/国储局Ⅱ期/实验/guochushiyan/winequality/wti-senti-week-hp-g.csv"
}
table <- read.csv(infile, sep=",", header=T)
senti = table[["sentiment"]]
wti = table[["wti"]]
date = table[["date"]]

sentits<-ts(senti,frequency = 12,start=c(2013,7))
wtits<-ts(wti,frequency = 12,start=c(2013,7))

opar <- par(no.readonly=TRUE)
library(mFilter)
senti.hp <- hpfilter(sentits)
wti.hp <- hpfilter(wtits)
plot(senti.hp)
plot(wti.hp)
par(opar)

d <- data.frame(date,wti.hp$trend,senti.hp$trend)
names(d) <- c("date","wti","senti")
write.csv(d, file = outfile, row.names = F, quote = F)