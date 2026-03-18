
s = 0.1;

Point(1) = {0, 0, 0, s};
Point(2) = {1, 0, 0, s};
Point(3) = {1, 1, 0, s};
Point(4) = {0, 1, 0, s};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};


Physical Curve("sides") = {1, 2, 3, 4};
Physical Surface("internal") = {1};

//Mesh.MeshSizeFromPoints = 1;


Transfinite Curve{1, 2, 3, 4} = 41;
Transfinite Surface{1};

Recombine Surface{1};

