#include<stdio.h>
int add(int*,int*);
int main()
{
    int a,b;
    printf("\n Enter the number=");
    scanf("%d%d",&a,&b);
    int sum =add(&a,&b);
    printf("addition is:%d",sum);
    return 0;

    
}
int add(int *x,int *y)
    {
        int z = *x + *y ;
        return z;
    }
