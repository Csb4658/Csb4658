#include<stdio.h>
#include<stdlib.h>
typedef struct
{
int pid,btime,wtime,ttime;
}sp;
int main()
{
int i,j,n,tbm=0,totwtime=0,tottime;
sp*p,t;
printf("sjf scheduling\n");
printf("enter the no.of processes:");
scanf("%d",&n);
p=(sp*)malloc(n*sizeof(sp));
printf("\n enter the burst time for each process:\n");
for(i=0;i<n;i++)
{
printf("process %d:",i+1);
scanf("%d",&p[i].btime);
p[i].pid=i+1;
p[i].wtime=0;
}
for(i=0;i<n;i++)
{
for(j=i+1;j<n;j++)
if(p[i].btime>p[j].btime)
{
t=p[i];
p[i]=p[j];
p[j];
}
}
}
printf("\n process scheduling..."\n);
printf("process\tbursttime\twaiting time\t turnaround time..."\n);
for(i=0;i<n;i++)
{
totwtime+=p[i].wtime=tbm;
p[i].ttime=p[i].wtime+p[i].btime;
tbm+=p[i].btime;
}
tottime=tbm+totwtime;
printf("\n total waiting time:%d...",totwtime);
printf("\n avg waiting time:%f...",(float)totwtime\n);
printf("\n total turnaround time:%d...",tottime);
printf("\n avg turnaround time:%f...",(float)tottime\n);
free(p);
return 0;
}

