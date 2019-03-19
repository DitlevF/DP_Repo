%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This code produces a shape-preserving approximation to fhandle by   %%%
%%% Schumaker's (1983) pice-wise cubic spline. The algorithm is based   %%% 
%%% on Judd (1998, Algorithm 6.3, p.233)                                %%%
%%% By: Thomas Jørgensen, d.22/9-2010                                   %%%
%%% Updated 23/9-2010: Butlan (1980)-slopes, from Iqbal (1994, eq. 4.1) %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT ARGUMENTS (notation follows Judd (1998)):
%             t:                     The nx1 vector of nodes/knots
%             z:                     The nx1 vector of function values
%             xv:                    The mx1 vector of evaluation points
% 
% OUTPUT ARGUMENTS:
%             yv:                    The mx1 vector of approximated function values
%             s:                     The nx1 vector of slopes at the nodes/knots
%             d:                     The n-1x1 vector of delta's
%             zeta:                  The n-1x1 vector of ekstra notes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [yv, s, d, zeta]=SchumakerSpline(t,z,xv)
%This is the main function. The subfunctions are ordered underneath in
%order of appearance
n=size(t,1);                    % Number of nodes/knots in the grid
error=CheckInput(t,n,xv);          % Check whether the inputs are okay

% Compute the slope (s) and the ekstra nodes (zeta)
[s, d] = CompSlope(z,t,n);
zeta = CompZeta(t,s,d,n);

% Evaluate the Spline-approximation at xv
dim = size(xv,1);       % # of rows/ evaluation points in xv
yv = zeros(dim,1)+NaN;
for eval=1:1:dim
    yv(eval)=EvalSpline(t,z,n,s,zeta,xv(eval));
end

end

function err=CheckInput(t,n,xv)
%This function merely checks the input
err=0;
if n<3
    err = 1;
    error('ERROR: To few grid-points. Need at least three points in order to interpolate!');
end
for i=1:n-1
    if t(i)>=t(i+1)                                                         % Checks if the grids are not sorted in descending order
        err = 2;
        error('ERROR: The grids are not sorted in descending order!');
    end
end
if size(xv,2)>1
    err = 3;
    error('ERROR: The points of evaluation (xv), has to be a mx1 vector!');
end
end

function [s, d]=CompSlope(z,t,n)
% This function computes the slope in the different nodes/knots
s=zeros(n,1);
L=zeros(n-1,1);
d=zeros(n-1,1);
for i=1:n-1;
    % Calculate the slopes
    L(i) = sqrt((t(i+1)-t(i))^2 + (z(i+1)-z(i))^2); 
    d(i) = (z(i+1)-z(i))/(t(i+1)-t(i));
    if i>1
        if d(i-1)*d(i)>0
%             s(i) = (L(i-1)*d(i-1)+L(i)*d(i)) / (L(i-1)+L(i));
            s(i) = (2*d(i-1)*d(i))/(d(i-1)+d(i));                            % This is the Butlan (1980)-slopes, from Iqbal (1994, eq. (4.1), p.202)
        else 
            s(i) = 0;
        end;
    end;
end;
% Compute the slope at the end-points
s(1) =  (3*d(1)-s(2))/2;            % Is this the right sign??
s(n) =  (3*d(n-1)-s(n-1))/2;
end

function [zeta] = CompZeta(t,s,d,n)
% This function computes the free node, zeta, based on the slopes at the
% nodes/knots
zeta = zeros(n-1,1);         % Initial matrix to store the eta's in
for i=1:n-1
    delta1 = s(i)-d(i);
    delta2 = s(i+1)-d(i);
    if (delta1*delta2)>=0;                                                  % Then there must be an inflection point in [t(i);t(i+1)]
        zeta(i) = (t(i)+t(i+1))/2;                                           % Choose the average of t(i) and t(i+1)     
    elseif abs(delta1) > abs(delta2);                                         % Then the function is either convex or concave
        zeta_bar = t(i) + 2*(t(i+1)-t(i))*delta2 / (s(i+1)-s(i));     % eta_bar from eq. (6.11.4)
        zeta(i) = (t(i)+zeta_bar)/2;                                          % Choose the average of t(i) and zeta_bar    
    else
        zeta_bar = t(i+1) + 2*(t(i+1)-t(i))*delta1 / (s(i+1)-s(i));    % eta_bar from eq. (6.11.5)
        zeta(i) = (zeta_bar+t(i+1))/2;                                       % Choose the average of zeta_bar and t(i+1)  
    end;
end;
end

function yv=EvalSpline(t,z,n,s,zeta,xv)
% This function evaluates the spline at xv and returns the function-value yv

% Need to find which interval (i) and left or rigth of zeta (j), in order to compute eq. (6.11.3)
for i=1:n-1
    if zeta(i)>=xv
        j=1;
        break
    elseif t(i+1)>=xv
        j=2;
        break
    elseif i==n-1
        if zeta(i)>=xv
            j=1;
            break
        else
            j=2;
            break
        end;
    end;
end;

% Evaluates the function in the given interval found above (i,j)
A1 = z(i);
B1 = s(i);
alpha = zeta(i) - t(i);
beta = t(i+1) - zeta(i);
s_bar = (2*(z(i+1)-z(i))-(alpha*s(i)+beta*s(i+1))) / (t(i+1)-t(i));
C1 = (s_bar-s(i)) / (2*alpha);      

if j==1
    yv = A1 + B1*(xv-t(i)) + C1*(xv-t(i))^2;                                %Eq. (6.11.3)
else
    A2 = A1 + alpha*B1 + (alpha^2)*C1;
    B2 = s_bar;
    C2 = (s(i+1)-s_bar) / (2*beta);
    yv = A2 + B2*(xv-zeta(i)) + C2*(xv-zeta(i))^2;
end;

end