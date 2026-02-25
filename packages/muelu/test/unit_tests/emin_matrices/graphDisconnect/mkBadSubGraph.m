%%  Edges marked with dashes are strong. Edges marked with dots are weak.
%%  Edges marked with plus signs were originall supposed to be weak edges
%%  (based on my guess of how MueLu picks root nodes). However, my guess
%%  was wrong, and so these ones are now set as strong. The main idea is
%%  that we want four perfect 2x2 aggregates to be chosen in each corner. 
%%  Getting this depends a bit on how MueLu chooses root nodes. The main
%%  idea is that SW aggregate has a strong connection to the NE aggregate.
%%  But since there are no fine edges connecting these two aggregagtes,
%%  a D0coarse edge will not be created between them. The same is also
%%  true for the interaction between the NW and SE aggregates. Thus,
%%  D0coarse consists of 4 edges that for the box that connects the 
%%  four aggregates.  However, the Pn basis function associated with 
%%  the SW aggregate has support within the NE aggregate (likewise for 
%%  the relationship between the NW and SE basis functions). For example,
%%  Dfine(4,:)*Pn is nonzero for coarse nodes 1 and 4 (assuming they
%%  are numbered lexicographically). Here, the 4th edge
%%  is the horizontal edge that connects node 5 with 6. Again, this is 
%%  due to the strong connection between the two associated aggregates.
%%  But since there is no strong connection to aggregate 2 and 3, 
%%  these do not appear in Dfine(4,:)*Pn . As there is no coarse edge
%%  between nodes 1 and 4, we can not solve the commuting relationship
%%
%%  Notice if there had been a strong connection between from either
%%  fine node 5 or 6 to a fine node in aggregate 3, then Dfine(4,:)*Pn 
%%  would have also included
%%  coarse node 3. In this scenario, it would be possible to solve
%%  this computing equation as the Dcoarsse path (1 --> 2) and
%%  (2-->3) would in fact lead to a connected sub-graph that includes
%%  coarse nodes 1, 3, and 4. 
%%
%%                 13          14          15          16
%%                  r --------- o --------- o --------- r
%%                  | \       / .           + \       / |
%%                  |   \   /   .           +   \   /   |
%%                  |     x     .           +     x     |
%%                  |   /   \   .           +   /   \   |
%%                  | /       \ .           + /       \ |
%%                9 o + + + + + o . . . . . o + + + + + o 12
%%                  .           . \       / .           .
%%                  .           .   \   /   .           .
%%                  .           .     x     .           .
%%                  .           .   /   \   .           .
%%                  .           . /       \ .           .
%%                5 o . . . . . o . . . . . o . . . . . o 8
%%                  | \       / .           + \       / |
%%                  |   \   /   .           +   \   /   |
%%                  |     x     .           +     x     |
%%                  |   /   \   .           +   /   \   |
%%                  | /       \ .           + /       \ |
%%                  r --------- o --------- o --------- r
%%                  1           2           3           4
%%
eps = -.00001;
fixed = -1;
An = spalloc(16,16,16*9); 
%          horizontal edges
An( 1, 2) = -1.; An( 2, 3) = eps; An( 3, 4) = -1.;
An( 5, 6) = eps; An( 6, 7) = eps; An( 7, 8) = eps;
An( 9,10) =fixed;An(10,11) = eps; An(11,12) = fixed;
An(13,14) = -1.; An(14,15) = -1.; An(15,16) = -1.;

%          vertical   edges
An( 1, 5) = -1.; An( 2, 6) = eps; An( 3, 7) = 1.; An( 4, 8) = -1.;
An( 5, 9) = eps; An( 6,10) = eps; An( 7,11) = eps; An( 8,12) = eps;
An( 9,13) = -1.; An(10,14) = eps; An(11,15) = fixed; An(12,16) = -1.;

%             /  edges
An( 1, 6) = -1.; An( 3, 8) = -1.;
An( 6,11) = -1.; 
An( 9,14) = -1.; An(11,16) = -1.;

%             \  edges
An( 2, 5) = -1.; An( 4, 7) = -1.;
An( 7,10) = -1.; 
An(10,13) = -1.; An(12,15) = -1.;
An = An + An';
An = An + spdiags(-(An*ones(16,1)),0,16,16);

Dfine = spalloc(24,16,24*2); 
% horizontal edges
Dfine( 1, 1) = -1; Dfine( 1, 2) = 1;    Dfine( 2, 2) = -1; Dfine( 2, 3) = 1;    Dfine( 3, 3) = -1; Dfine( 3, 4) = 1; 
Dfine( 4, 5) = -1; Dfine( 4, 6) = 1;    Dfine( 5, 6) = -1; Dfine( 5, 7) = 1;    Dfine( 6, 7) = -1; Dfine( 6, 8) = 1; 
Dfine( 7, 9) = -1; Dfine( 7,10) = 1;    Dfine( 8,10) = -1; Dfine( 8,11) = 1;    Dfine( 9,11) = -1; Dfine( 9,12) = 1; 
Dfine(10,13) = -1; Dfine(10,14) = 1;    Dfine(11,14) = -1; Dfine(11,15) = 1;    Dfine(12,15) = -1; Dfine(12,16) = 1; 

% vertical edges

e = 13;
Dfine( e, 1) = -1; Dfine( e, 5) = 1;    Dfine(e+1, 2) = -1; Dfine(e+1, 6) = 1;    Dfine(e+2, 3) = -1; Dfine(e+2, 7) = 1;     Dfine(e+3, 4) = -1; Dfine(e+3, 8) = 1; e=e+4;
Dfine( e, 5) = -1; Dfine( e, 9) = 1;    Dfine(e+1, 6) = -1; Dfine(e+1,10) = 1;    Dfine(e+2, 7) = -1; Dfine(e+2,11) = 1;     Dfine(e+3, 8) = -1; Dfine(e+3,12) = 1; e=e+4;
Dfine( e, 9) = -1; Dfine( e,13) = 1;    Dfine(e+1,10) = -1; Dfine(e+1,14) = 1;    Dfine(e+2,11) = -1; Dfine(e+2,15) = 1;     Dfine(e+3,12) = -1; Dfine(e+3,16) = 1; e=e+4;

Ae = spalloc(24,24,7*24);
for i=1:3, for j= 1:3,  
   hor = (j-1)*3 + i; ver = 12+(j-1)*4+i;
   Ae(hor  ,hor  ) = Ae(hor  ,hor  ) + 1;
   Ae(hor+3,hor+3) = Ae(hor+3,hor+3) + 1;
   Ae(ver  ,ver  ) = Ae(ver  ,ver  ) + 1;
   Ae(ver+1,ver+1) = Ae(ver+1,ver+1) + 1;
   Ae(hor  ,hor+3) = Ae(hor  ,hor+3) - 1;
   Ae(hor+3,hor  ) = Ae(hor+3,hor  ) - 1;
   Ae(ver  ,ver+1) = Ae(ver  ,ver+1) - 1;
   Ae(ver+1,ver  ) = Ae(ver+1,ver  ) - 1;

   Ae(hor  ,ver  ) = Ae(hor  ,ver  ) - 1;       Ae(ver  ,hor  ) = Ae(ver  ,hor  ) - 1;
   Ae(hor  ,ver+1) = Ae(hor  ,ver+1) -    (-1); Ae(ver+1,hor  ) = Ae(ver+1,hor  ) -    (-1);
   Ae(hor+3,ver  ) = Ae(hor+3,ver  ) -    (-1); Ae(ver  ,hor+3) = Ae(ver  ,hor+3) -    (-1); 
   Ae(hor+3,ver+1) = Ae(hor+3,ver+1) - 1 ;      Ae(ver+1,hor+3) = Ae(ver+1,hor+3) -1 ; 
end;end;

mmwrite('A.dat',Ae);
mmwrite('An.dat',An);
mmwrite('D0h.dat',Dfine);



