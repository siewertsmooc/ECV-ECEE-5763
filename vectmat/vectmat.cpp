// Note this example is not thread safe - should only be called by single thread main()
//
// This code demonstrates both bad practices as well as good, so read it carefully.
//
// Later in the course, we will learn about memory management, overloading operators, and
// thread safety to improve this, with the ultimate goal to create a vector/matrix library of
// functions that could be used safely and coneniently so that it matches textbook math well.
//
// Explore how this example code works in your debugger and create a new version of the overloaded
// functions for the sv3_t and sm3_t with the same names as the original vector3 and matrix3 versions.

#include <iostream>
#include <cmath>
#include <stdlib.h>

using namespace std;

// we can define a vector like this
typedef struct
{
	float x;
	float y;
	float z;
} svector3;


// Note that we can create a derived type that encapsulates a 3D vector or
// matrix inside a structure.
typedef struct
{
	float v3[3];
} sv3_t;

typedef struct
{
	float m3[3][3];
} sm3_t;


// or, if we use an array, here's some basic three space vector matrix algebra types
typedef float vector3[3];
typedef float matrix3[3][3];
typedef float scalar;

// More advanced prototypes we may want to fill in bodies for later ...
vector3& normalize(vector3& v);
scalar magnitude(vector3 v);
vector3& scale(vector3& v, scalar s);
scalar dotprod(vector3 v1, vector3 v2);
vector3& crossprod(vector3 v1, vector3 v2);
matrix3& product(matrix3 m1, matrix3 m2);
vector3& rotate(matrix3 m, vector3 v);

// vector3 add(vector3 v1, vector3 v2);   -- would be nice if we could return a vector, but we can't 
vector3& add(vector3 v1, vector3 v2);  // -- we can return a reference to a vector, is this what we want?
vector3& add3(vector3 v1, vector3 v2, vector3 v3); //    -- ambiguous overload? yes
// This might be safer, but harder to nest in a f(g(f)) type of way
void add(vector3& v1, vector3& v2, vector3& vsum); //    -- ambiguous overload? 
void add(vector3 v1, vector3 v2, vector3 * vsum);  //    -- ambiguous overload? no, different type signature then version of add using references
vector3* add_with_alloc(vector3 v1, vector3 v2);   // who free's the memory that is allocated?

void coutv(vector3 v);
void coutv(vector3 *vptr);
void coutv(svector3 sv);

void coutm(matrix3 m);

// function prototypes for new coutv and coutm for sv3_t and sm3_t
void coutv(sv3_t v3);
void coutm(sm3_t m3);


int main(void)
{
	scalar s1=1.0, s2=2.0;
	vector3 v1={1.0,0.0,0.0}, v2={0.0,1.0,0.0}, v3={0.0,0.0,1.0}, vsum;
	vector3 vbig={10.0, 10.0, 10.0};
	vector3 vtemp={0.0,0.0,0.0};
	vector3* vptr; // uninitialized pointer!

	matrix3 m1={{1.0,2.0,3.0},{1.0,2.0,3.0},{1.0,2.0,3.0}};
	matrix3 m2={{1.0,1.0,1.0},{2.0,2.0,2.0},{3.0,3.0,3.0}};

	matrix3 rotx={{1.0,0.0,0.0},{0.0,0.0,-1.0},{0.0,1.0,0.0}};
	matrix3 roty={{0.0,0.0,1.0},{0.0,1.0,0.0},{-1.0,0.0,0.0}};
	matrix3 rotz={{0.0,-1.0,0.0},{1.0,0.0,0.0},{0.0,0.0,1.0}};

	svector3 sv1={1.0,0.0,0.0}, sv2={0.0,1.0,0.0}, sv3;

	// structure encapsulated vector and matrix declarations -- note more complex initializers
	sv3_t sv3_x={{1.0,0.0,0.0}}, sv3_y={{0.0,1.0,0.0}}, sv3_z={{0.0,0.0,1.0}};
	sm3_t sm3_rotx={{{1.0,0.0,0.0},{0.0,0.0,-1.0},{0.0,1.0,0.0}}};
	sm3_t sm3_roty={{{0.0,0.0,1.0},{0.0,1.0,0.0},{-1.0,0.0,0.0}}};
	sm3_t sm3_rotz={{{0.0,-1.0,0.0},{1.0,0.0,0.0},{0.0,0.0,1.0}}};

	sv3_t sv3_a;
	sm3_t sm3_a;


	//v3 = v2;  -- would be nice if we could do this, but v3 is not a valid l-value
	// v3 = {0.0,0.0,1.0}; -- nor this
	cout << "array vector3 v3="; coutv(v3); cout << endl;

	sv3=sv2;
	// sv3={0.0,0.0,1.0}; -- still can't do this
	cout << "svector3 sv3="; coutv(sv3); cout << endl;

	// now, what if we use assignment with the structure encapsulated types?
	// can we in fact assign an instance of the same type that is pre-initialized to
	// an uninitialized version of that type?
	sv3_a = sv3_x;
	sv3_a = sv3_y;
	sm3_a = sm3_rotx;
	sm3_a = sm3_rotx;

	cout << "\nScalar Addition:\n";
	cout << "s1=" << s1 << ", s2=" << s2 << ": s1 + s2=" << (s1+s2) << endl;

	cout << "\nVector Addition:\n";
	// cout << "v1=" << v1 << ", v2=" << v2 << ": v1 + v2=" << (v1+v2) << endl;     -- would be nice, but + not defined for our type
	cout << "v1=" << v1 << ", v2=" << v2 << ": v1 + v2=" << add(v1, v2) << endl; // -- prints addresses, not contents like we want
	cout << "v1="; coutv(v1); cout << ", v2="; coutv(v2); cout << ", v3="; coutv(v3); cout << ": v1 + v2 + v3="; coutv(add(add(v1, v2), v3)); cout << endl;
	cout << "v1="; coutv(v1); cout << ", v2="; coutv(v2); cout << ", v3="; coutv(v3); cout << ": v1 + v2 + v3="; coutv(add3(v1, v2, v3)); cout << endl;

	// This version of add uses 2 pass by reference array inputs and one reference for update
	cout << "v1="; coutv(v1); cout << ", v2="; coutv(v2); cout << ": v1 + v2="; add(v1, v2, vsum); coutv(vsum); cout << endl;

	// This version uses 2 pass by reference array inputs and one pointer to an array
	cout << "v1="; coutv(v1); cout << ", v2="; coutv(v2); cout << ": v1 + v2="; add(v1, v2, &vsum); coutv(vsum); cout << endl;

	cout << "\nVector & Matrix Operations:\n";
	cout << "v="; coutv(vbig); cout << "mag(v)=" << magnitude(vbig); cout << ", vnorm="; coutv(normalize(vbig)); cout << endl;
	cout << "v1="; coutv(v1); cout << ", v2="; coutv(v2); cout << ", and v1 dot v2=" << dotprod(v1, v2); cout << endl;
	cout << "v1="; coutv(v1); cout << ", v2="; coutv(v2); cout << ", and v1 X v2="; coutv(crossprod(v1, v2)); cout << endl;
	cout << "m1="; coutm(m1); cout << ", m2="; coutm(m2); cout << ", and m1 * m2="; coutm(product(m1, m2)); cout << endl;
	cout << "v1="; coutv(v1); cout << ", roty="; coutm(roty); cout << ", and v1 * roty="; coutv(rotate(roty, v1)); cout << "\n\n";

	cout << "v1="; coutv(v1); cout << ", v2="; coutv(v2); cout << ": v1 + v2="; coutv(add(v1, v2)); cout << endl;

	cout << "\nEntering loop with allocating vector addition\n";
	{
		int idx; // declaration in local block

		for(idx=0; idx < 10; idx++)
		{
			vptr=add_with_alloc(v1, v2);

			// if you comment out the free, which returns memory allocated, your
			// application will likely be halted with an exception.
			free(vptr);

			if((idx % 100000) == 0) cout << ".";
		}
		
		cout << "\nv1="; coutv(v1); cout << ", v2="; coutv(v2); cout << ": v1 + v2="; 
		coutv(vptr=add_with_alloc(v1, v2)); cout << " called " << idx << " times with allocation" << endl;
	}

	cout << "\nDone\n";

	return 0;
}

scalar add(scalar s1, scalar s2)
{
	return (s1+s2);
}

// can we return a reference to vector? yes, but be careful how
static vector3 vsum2;
vector3& add(vector3 v1, vector3 v2)
{
	// vector3 vsum2;

	vsum2[0]=v1[0]+v2[0];
	vsum2[1]=v1[1]+v2[1];
	vsum2[2]=v1[2]+v2[2];

	return(vsum2);
}


void add(vector3 v1, vector3 v2, vector3 * vsum)
{
	for(int idx=0; idx < 3; idx++)
		(*vsum)[idx]=v1[idx]+v2[idx];
}



void add(vector3 & v1, vector3 & v2, vector3 & vsum)
{
	for(int idx=0; idx < 3; idx++)
		vsum[idx]=v1[idx]+v2[idx];
}


static vector3 vsum3;
vector3& add3(vector3 v1, vector3 v2, vector3 v3)
{

	vsum3[0]=v1[0]+v2[0]+v3[0];
	vsum3[1]=v1[1]+v2[1]+v3[1];
	vsum3[2]=v1[2]+v2[2]+v3[2];

	return(vsum3);
}


vector3* add_with_alloc(vector3 v1, vector3 v2)
{
	vector3 *vptr;

	vptr=(vector3 *)malloc(3*sizeof(float));  // we are counting on someone freeing this in the future, or creating memory leak

	for(int idx=0; idx < 3; idx++)
		(*vptr)[idx]=v1[idx]+v2[idx];

	return vptr;
}


void coutv(vector3 v)
{
	for(int idx=0; idx < 3; idx++)
		cout << v[idx] << " ";
}


void coutv(vector3 *vptr)
{
	for(int idx=0; idx < 3; idx++)
		cout << (*vptr)[idx] << " ";
}


void coutm(matrix3 m)
{
	for(int row=0; row < 3; row++)
	{
		for(int col=0; col < 3; col++)
			cout << m[row][col] << " ";

		cout << ";  ";
	}
}


void coutv(svector3 sv)
{
	cout << sv.x << " " << sv.y << " " << sv.z;
}


// Implement these and add new overloaded operations on these types as well
void coutv(sv3_t v3)
{
	cout << "Replace me with a new output method\n";
}

void coutm(sm3_t m3)
{
		cout << "Replace me with a new output method\n";
}


// Original test fucntions on the typedef array type
//
static vector3 normv;
vector3& normalize(vector3& v)
{
	float mag=magnitude(v);

	for(int idx=0; idx < 3; idx++)
		normv[idx]=v[idx]/mag;

	return(normv);
}

scalar magnitude(vector3 v)
{
	return(sqrt((v[0]*v[0]) + (v[1]*v[1]) + (v[2]*v[2])));
}

static vector3 scalev;
vector3& scale(vector3& v, scalar s)
{
	for(int idx=0; idx < 3; idx++)
		scalev[idx]=v[idx]*s;

	return(scalev);
}

// returns zero if 2 vectors are orthogonal, 
// otherwise dotprod is magnitude(v1)*magnitude(v2)*cos(angle-between-them)
scalar dotprod(vector3 v1, vector3 v2)
{
	return ((v1[0]*v2[0]) + (v1[1]*v2[1]) + (v1[2]*v2[2]));
}


// Cross product of v1 X v2 is:
// = (v1[0]i + v1[1]j + v1[2]k) X (v2[0]i + v2[1]j + v2[2]k)
//
// This simplifies where (i X i) = 0, (j X j) = 0, (k X k) = 0
//
static vector3 v1xv2;
vector3& crossprod(vector3 v1, vector3 v2)
{
	// scalar components for v1xv2 =c1*i + c2*j + c3*k
	v1xv2[0]=v1[1]*v2[2] - v1[2]*v2[1];
	v1xv2[1]=v1[2]*v2[0] - v1[0]*v2[2];
	v1xv2[2]=v1[0]*v2[1] - v1[1]*v2[0];

	return v1xv2;
}

// Multiply each row * column and sum
// A 3x3 * 3x3 yields a 3x3
// where element(0,0) = row0 * col0 and so on ... 
static matrix3 m1xm2;
matrix3& product(matrix3 m1, matrix3 m2)
{
	int row, col;

	for(row=0; row < 3; row++)
		for(col=0; col < 3; col++)
			m1xm2[row][col]=0.0;

	for(row=0; row < 3; row++)
	{
		for(col=0; col < 3; col++)
		{
			for(int idx=0; idx < 3; idx++)
			{
				m1xm2[row][col]+=m1[row][idx]*m2[idx][col];
			}
		}
	}

	return m1xm2;
}

// Multiply each row * column and sum
// A 1x3 * 3x3 yields a rotated 1x3
// where element(0,0) = row0 * col0 and so on ... 
static vector3 rv;
vector3& rotate(matrix3 m, vector3 v)
{
	int col;

	for(col=0; col < 3; col++)
		rv[col]=0.0;

	for(col=0; col < 3; col++)
	{
		for(int idx=0; idx < 3; idx++)
		{
			rv[col]+=m[idx][col]*v[idx];
		}
	}

	return rv;
}
