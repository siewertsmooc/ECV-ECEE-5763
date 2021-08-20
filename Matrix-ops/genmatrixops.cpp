// https://www.programming-techniques.com/2012/03/3d-transformation-translation-rotation-and-scaling-in-c-c.html
//
#include <iostream>
#include <cmath> 
 
using namespace std; 
 
typedef struct {
    float x;
    float y;
    float z;
}Point;
Point points; 
 
float temp = 0; 
 
void showPoint(){
    cout<<"("<<points.x<<","<<points.y<<","<<points.z<<")"<<endl;
} 
 
void translate(float tx, float ty, float tz){
    points.x += tx;
    points.y += ty;
    points.z += tz;
    cout<<"After Translation, new point is :";
    showPoint();
} 
 
void rotatex(float angle){
    angle = angle * M_PI / 180.0;
    temp = points.y;
    points.y = points.y * cos(angle) - points.z * sin(angle);
    points.z = temp * sin(angle) + points.z * cos(angle);
    cout<<"After rotation about x, new point is: ";
    showPoint();
} 
 
void rotatey(float angle){
    angle = (angle * M_PI) / 180.0;
    temp = points.z;
    points.z = points.z * cos(angle) - points.x * sin(angle);
    points.x = temp * sin(angle) + points.x * cos(angle);
    cout<<"After rotation about y, new point is: ";
    showPoint(); 
 
} 
 
void rotatez(float angle){
    angle = angle * M_PI / 180.0;
    temp = points.x;
    points.x = points.x * cos(angle) - points.y * sin(angle);
    points.y = temp * sin(angle) + points.y *cos(angle);
    cout<<"After rotation about z, new point is: ";
    showPoint(); 
 
} 
 
void scale(float sf, float xf, float yf, float zf){
    points.x = points.x * sf + (1 - sf) * xf;
    points.y = points.y * sf + (1 - sf) * yf;
    points.z = points.z * sf + (1 - sf) * zf;
    cout<<"After scaling, new point is: ";
    showPoint();
} 
 
int main()
{
    float tx = 0, ty = 0, tz = 0;
    float sf = 0, xf = 0, yf = 0, zf = 0;
    int choose;
    float angle;
    cout<<"Enter the initial point you want to transform:";
    cin>>points.x>>points.y>>points.z;
    cout<<"Choose the following: "<<endl;
    cout<<"1. Translate"<<endl;
    cout<<"2. Rotate about X axis"<<endl;
    cout<<"3. Rotate about Y axis"<<endl;
    cout<<"4. Rotate about Z axis"<<endl;
    cout<<"5. Scale"<<endl;
    cin>>choose;
    switch(choose){
        case 1:
            cout<<"Enter the value of tx, ty and tz: ";
            cin>>tx>>ty>>tz;
            translate(tx, ty, tz);
            break;
        case 2:
            cout<<"Enter the angle: ";
            cin>>angle;
            rotatex(angle);
            break;
        case 3:
            cout<<"Enter the angle: ";
            cin>>angle;
            rotatey(angle);
            break;
        case 4:
            cout<<"Enter the angle: ";
            cin>>angle;
            rotatez(angle);
            break;
        case 5:
            cout<<"Enter the value of sf, xf, yf and zf: ";
            cin>>sf>>xf>>yf>>zf;
            scale(sf, xf, yf, zf);
            break;
        default:
            break;
    }
    return 0;
}
