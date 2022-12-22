## Generate data from a multivariate normal

library(MASS)


p_1 = 0.4
p_2 = 0.4
p_3=0.2


mu_1=c(1,1)
sigma1=matrix(c(1,0,0,1),ncol=2,byrow=TRUE)

mu_2=c(5,10)
sigma2=matrix(c(2,0.5,0.5,2),ncol=2,byrow=TRUE)


n=100

#library(MASS)
c1=matrix(c(NA,NA),ncol=2,nrow=n)
c2=matrix(c(NA,NA),ncol=2,nrow=n)

prob=vector(length=n)
for(i in 1:n){
  prob[i]=runif(1,0,1)
  if (prob[i]<=0.4) {
    c1[i,] = mvrnorm(1,mu_1,sigma1)
  } else{
    c2[i,] = mvrnorm(1,mu_2,sigma2)
  }
}

c1v2=as.data.frame(c1[complete.cases(c1),])
c2v2=as.data.frame(c2[complete.cases(c2),])


c1v2$group=rep("0",nrow(c1v2))
c2v2$group=rep("1",nrow(c2v2)) 

datasim=rbind(c1v2,c2v2)
plot(datasim[,1:2],col=datasim$group)

write.csv(datasim$group,"trueclass.csv")
write.csv(datasim[,1:2],"input_sim_data.csv")


