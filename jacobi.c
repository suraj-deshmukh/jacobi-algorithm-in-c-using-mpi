#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<math.h>


void jacobi(float *X,float *N_x,float **a,int *diag,int Count,int N,int Column,int Rank)
{  
   //N=total no of elements in A
   int i=0,j=0,k=0,n,c=Column-1;
   n=N/Count;
   float sum1=0,sum2=0;
   i=0;
   while(i<Count)
   {
      for(j=0;j<diag[i];j++)
         sum1=sum1+a[i][j]*X[j];
      for(j=(diag[i]+1);j<c;j++)
         sum2=sum2+a[i][j]*X[j];
      N_x[i]=(a[i][c] - sum1 - sum2)/a[i][diag[i]];
      i++; 
      sum1=0;
      sum2=0;  
   }
}


int main(int argc,char *argv[])
{
  int rank,size,*sendcount,*displace,*reccount,*rowcount,*rowdisplace,*diagonal,*Diag;  //rowcount = sendcount for diag Diag=for each proc
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&size); 
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Status status;
  FILE *fp;
  char c;
  int i,j,k=0,count=0,row=0,column=0,itr=0;
  float n=0,*sum,*rec_data,*data,*x,*new_x,*temp,eps=0.00001,norm1=0,norm2=0;
  sendcount = (int*)calloc(sizeof(int),size);
  rowcount = (int*)calloc(sizeof(int),size);
  rowdisplace = (int*)calloc(sizeof(int),size);
  reccount = (int*)calloc(sizeof(int),size);
  displace = (int*)calloc(sizeof(int),size);
  if(rank==0)
  {
    fp=fopen("matrix.txt","r");
    while(fscanf(fp,"%f",&n)!=-1)
    { 
      c=fgetc(fp);
      if(c=='\n'){ row=row+1; }
      count++;
    }
    if(size>row){ printf("No of proc are greater than no of row.\nCode Terminated.\n");exit(0);}
    column=count/row; 
    printf("Row=%d column=%d proc=%d\n",row,column,size);
    fseek( fp, 0, SEEK_SET );
    data=(float*)calloc(sizeof(float),row*column);
    x=(float*)calloc(sizeof(float),row);  //x = initial guess
    //diagonal=(int*)calloc(sizeof(int),row);  //x = initial guess
    for(i=0;i<row;i++) 
    {
       for(j=0;j<column;j++)
       {
          fscanf(fp,"%f",&n);
          data[k]=n;
          k++; 
       }
       diagonal[i]=i; 
       //printf("diagonal=%d\n",diagonal[i]);
    }
    fclose(fp);
    fp=fopen("guess.txt","r"); 
    count=0;
    fseek( fp, 0, SEEK_SET );
    for(i=0;i<row;i++)
      fscanf(fp,"%f",&x[i]); 
    fclose(fp);
    count=0;
    while(1)
    {
      for(i=0;i<size;i++)
      {
        sendcount[i] = sendcount[i]+1;
        rowcount[i] = sendcount[i]; 
        count++;
        if(count==row) break;  
      }
      if(count==row)  break; 
    }
    for(i=1;i<size;i++)
    {
      displace[i] = displace[i-1] + sendcount[i-1]*column;
      rowdisplace[i] = rowdisplace[i-1] + rowcount[i-1]; 
      sendcount[i-1] = sendcount[i-1] * column;
    }
    sendcount[size-1] = sendcount[size-1] * column;
    for(i=0;i<size;i++)
      printf("sendcout=%d rowcount=%d disp=%d rowdisp=%d\n",sendcount[i],rowcount[i],displace[i],rowdisplace[i]);
  }
  MPI_Bcast(sendcount,size,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(displace,size,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(rowcount,size,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(rowdisplace,size,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&row,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&column,1,MPI_INT,0,MPI_COMM_WORLD);
  if(rank!=0) x=(float*)calloc(sizeof(float),row);  //x = initial guess
  new_x=(float*)calloc(sizeof(float),row);  //new_x = new calculated x
  MPI_Bcast(x,row,MPI_INT,0,MPI_COMM_WORLD);
  Diag=(int*)calloc(sizeof(int),rowcount[rank]);
  rec_data=(float*)calloc(sizeof(float),sendcount[rank]);
  MPI_Scatterv(data,sendcount,displace,MPI_FLOAT,rec_data,sendcount[rank],MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Scatterv(diagonal,rowcount,rowdisplace,MPI_INT,Diag,rowcount[rank],MPI_INT,0,MPI_COMM_WORLD); 
  count=rowcount[rank];
  float *a[count];k=0;
  for(i=0;i<count;i++)
     a[i]=(float*)calloc(sizeof(float),column);
  temp=(float*)calloc(sizeof(float),row);
  for(i=0;i<count;i++)
  {
     for(j=0;j<column;j++)
       {
         a[i][j]=rec_data[k];
         k++;  
       }
  } 
  while(1)
  {
   jacobi(x,new_x,a,Diag,rowcount[rank],sendcount[rank],column,rank);
   MPI_Allgatherv(new_x,rowcount[rank],MPI_FLOAT,temp,rowcount,rowdisplace,MPI_FLOAT,MPI_COMM_WORLD);
   if(rank==0)
   {
    for(i=0;i<row;i++)
      {
        norm1=norm1+temp[i]*temp[i];
        norm2=norm2+x[i]*x[i];
      }
    norm1=sqrt(norm1);
    norm2=sqrt(norm2);
    if(fabs(norm1-norm2)<=eps)
    { 
      itr=1;
      printf("Final Ans\n");
      for(i=0;i<row;i++)
         printf("x[%d]=%.3f\n",i,x[i]);  
      free(diagonal);
    }
    norm1=0;
    norm2=0; 
   }
  MPI_Bcast(&itr,1,MPI_INT,0,MPI_COMM_WORLD);
  if(itr==1) break;
  for(i=0;i<row;i++)
    x[i]=temp[i];      
  }
  free(sendcount);
  free(displace);
  free(rowcount);
  free(displace);
  free(reccount);
  free(rec_data);
  free(Diag);
  free(x);
  free(new_x);
  free(*a);
  free(temp);
  MPI_Finalize();
  return 0;
}
