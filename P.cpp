
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
using namespace std;
#define MAX 5          //车库容量，可以根据实际情况改变
#define price 0.01     //一辆车每分钟费用，可变
 
typedef struct Time{
	int hour;
	int min;
}Time; //时间

typedef struct CarNode{
	char num[10];
	Time reach;
	Time leave;
}CarNode; //车辆信息

typedef struct Park{
	CarNode *stack[MAX + 1];
	int top;
}Park; //停车场

typedef struct CarLink{
	CarNode *data;
	struct CarLink *next;
}CarLink;

typedef struct Node{
	CarLink *head;
	CarLink *rear;
}Pave; //便道


void InitStack(Park *);                                 //初始化栈  
int InitQueue(Pave *);                                 //初始化便道 

int Arrival(Park *, Pave *);                           //车辆到达 
void Leave(Park *, Park *, Pave *);              //车辆离开 
void List(Park, Pave);                                  //显示信息 
void List1(Park *S);                                     //停车场信息
void List2(Pave *W);                                   //便道信息
void Print(CarNode *p, int room);              //输出离开车辆的信息

int main(){
	Park  S, Temp;
	Pave W;
	int ch;
	InitStack(&S); //初始化车站
	InitStack(&Temp); //初始化让路的临时栈
	InitQueue(&W); //初始化通道
	for(;;){ 
		system("cls");
		cout << "                                   @   欢迎使用停车管理系统   @                                      " << endl;
		cout << "**************************************************************************************************" << endl;
		cout << "                       ￥                     1. 车辆到达                 ￥                          " << endl;
		cout << "                       ￥                     2. 车辆离开                 ￥                          " << endl;
		cout << "                       ￥                     3. 信息显示                 ￥                          " << endl;
		cout << "                       ￥                     4. 退出                     ￥                          " << endl;
		cout << "**************************************************************************************************" << endl;
		cout << "请选择所需要的服务:                " << endl;
		while (1){
			cin >> ch;
			if (ch >= 1 && ch <= 4)break;
			else cout << "输入错误! 请重新选择：" << endl;
		}
		switch (ch){
		case 1:Arrival(&S, &W); break;                 //车辆到达 
		case 2:Leave(&S, &Temp, &W); break;    //车辆离开 
		case 3:List(S, W); break;                          //信息显示信息 
		case 4:exit(0);                                                   //退出主程序 
		default: break;
		}
	}
}

void InitStack(Park *s) {//初始化栈 
	int i;
	s->top = 0;
	for (i = 0; i <= MAX; i++)
		s->stack[s->top] = NULL;
}
int InitQueue(Pave *Q) {//初始化便道 
	Q->head = (CarLink *)malloc(sizeof(CarLink));
	if (Q->head != NULL){
		Q->head->next = NULL;
		Q->rear = Q->head;
		return(1);
	}
	else return(-1);
}

int Arrival(Park *S, Pave *W) {//车辆到达
	CarNode *p;
	CarLink *t;
	p = (CarNode *)malloc(sizeof(CarNode));
	cout << "请输入车牌号:" << endl;
	gets_s(p->num); 
	gets_s(p->num);
	if (S->top < MAX){  //若车场未满，车进车场
		S->top++;
		cout << "车辆在车场第 " << S->top << "号位置!" << endl;
		cout << "请输入到达时间: " << endl;
		cin >> p->reach.hour;
		if (p->reach.hour < 0 || p->reach.hour>23) {
			do {
				cout << "请输入正确的到达时间" << endl;
				cin >> p->reach.hour;
			} while (p->reach.hour <0 && p->reach.hour >23);
		}
		cin >> p->reach.min;
		if (p->reach.min < 0 || p->reach.min>59) {
			do {
				cout << "请输入正确的到达时间" << endl;
				cin >> p->reach.min;
			} while (p->reach.min <0 && p->reach.min >59);
		}
		S->stack[S->top] = p;
		return(1);
	}
	else{         //若车场已满，车辆进便道
		cout << "车场已满，请移至便道等待" << endl;
		t = (CarLink *)malloc(sizeof(CarLink));
		t->data = p;
		t->next = NULL;
		W->rear->next = t;
		W->rear = t;
		system("pause");
		return(1);	
	}
}

void Leave(Park *S, Park *Temp, Pave *W){ //车辆离开
	int room;
	CarNode *p;
	CarLink *q;	
	if (S->top > 0){ //判断车场内是否有车
		while (1){ 
			cout << "请输入车在停车场中的位置1--" << S->top << ":";
			cin >> room;
			if (room >= 1 && room <= S->top) break;
		}
		while (S->top > room){      //车辆离开 
			Temp->top++;          
			Temp->stack[Temp->top] = S->stack[S->top];
			S->stack[S->top] = NULL;
			S->top--;
		}
		p = S->stack[S->top];
		S->stack[S->top] = NULL;
		S->top--;
		while (Temp->top >= 1){  //判断临时通道上是否有车
			S->top++;
			S->stack[S->top] = Temp->stack[Temp->top];
			Temp->stack[Temp->top] = NULL;
			Temp->top--;
		}
		Print(p, room);
		if ((W->head != W->rear) && S->top < MAX){ //车站是未满,便道的车辆进入车场 
			q = W->head->next;
			p = q->data;  
			S->top++;
			cout << "便道的" << p->num << "号车进入车场第" << S->top << "号位置" << endl;
			cout << "请输入现在的时间如:" << endl;
			cin >> p->reach.hour;
			if (p->reach.hour < 0 || p->reach.hour>23) {
				do {
					cout << "请输入正确的到达时间" << endl;
					cin >> p->reach.hour;
				} while (p->reach.hour < 0 && p->reach.hour >23);
			}
			cin >> p->reach.min;
			if (p->reach.min < 0 || p->reach.min>59) {
				do {
					cout << "请输入正确的到达时间" << endl;
					cin >> p->reach.min;
				} while (p->reach.min < 0 && p->reach.min >59);
			}
			S->stack[S->top] = p;
			W->head->next = q->next;
			if (q == W->rear) W->rear = W->head;
			free(q); 
		}
		else {
			cout << "  当前便道没有车 " << endl;  //便道没车 
			system("pause");
		}
	}
	else {
		cout << "  当前停车场里没有车 " << endl; //车场没车 
		system("pause");
	}
}


void Print(CarNode *p, int room){ //输出离开车辆的信息清单 
	int A1, A2, B1, B2;
	cout << "请输入离开的时间:" << endl;
	cin >> p->leave.hour;
	if (p->leave.hour < 0 || p->leave.hour>23) {
		do {
			cout << "请输入正确的到达时间" << endl;
			cin >> p->leave.hour;
		} while (p->leave.hour < 0 && p->leave.hour >23);
	}
	cin >> p->leave.min;
	if (p->leave.min < 0 || p->leave.min>59) {
		do {
			cout << "请输入正确的到达时间" << endl;
			cin >> p->leave.min;
		} while (p->leave.min < 0 && p->leave.min >59);
	}
	cout << endl << "离开车辆的车牌号为: " << endl;
	puts(p->num);
	cout << "到达时间为: " << p->reach.hour << ":" << p->reach.min << endl;
	cout << "离开时间为: " << p->leave.hour << ":" << p->leave.min << endl;
	A1 = p->reach.hour;
	A2 = p->reach.min;
	B1 = p->leave.hour;
	B2 = p->leave.min;
	cout << "应交费用:   " << (((B1 - A1) * 60 + (B2 - A2)) + 1440) % 1440 * price << "元" << endl;
	free(p);
	system("pause");
}

void List(Park S, Pave W) {
	int  flag, kflag;
	flag = 1;
	cout << endl;
	while (flag) {
		cout << "请选择您要显示的信息 :" << endl;
		cout << endl;
		cout << "##   1.停车场  ##" << endl;
		cout << "##   2.便道    ##" << endl;
		cout << "##   3.返回    ##" << endl;
		while (1) {
			cin >> kflag;
			if (kflag >= 1 || kflag <= 3) break;
			else cout << "输入错误 请重新选择 :" << endl;
		}
		switch (kflag) {
		case 1:List1(&S); break;   //列表显示车场信息 
		case 2:List2(&W); break;   //列表显示便道信息 
		case 3:flag = 0; break;
		default: break;
		}
	}
}//列表界面 

void List1(Park *S){ 
	cout << endl;
	int i;
	if (S->top > 0){ //判断车站内是否有车
		cout << " 位置 到达时间 车牌号" << endl;
		for (i = 1; i <= S->top; i++){
			cout << "  " << i << "  " << S->stack[i]->reach.hour << ":" << S->stack[i]->reach.min << "  " << S->stack[i]->num << endl;
		}
		cout <<endl;
	}
	else {
		cout << "停车场里没有车" << endl;
		system("pause");
	}
} //列表显示车场信息

void List2(Pave *W){ 
	cout << endl;
	int i = 1;
	CarLink *p;
	p = W->head->next;
	if (W->head != W->rear){ //判断通道上是否有车
		cout << "等待车辆的车牌号码为:" << endl;
		while (p != NULL){
			cout << i << " " << p->data->num << endl;
			p = p->next;
			i++;
		} 
		cout << endl;
	}
	else {
		cout << "便道里没有车" << endl;
		system("pause");
	}
}//列表显示便道信息 

