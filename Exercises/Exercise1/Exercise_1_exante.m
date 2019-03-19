%% *Exercise 1*

clear
%% Brute force $T=2$
%%
% 1. setup
par       = struct();
par.alpha = 0.5;
par.beta  = 0.90;
par.T     = 2;
par.u     = @(x,par) x.^par.alpha;

% 2. state
M = 5;

% 3. solve
Vstar = -inf;
Cstar = nan(par.T,1);
for C1 = 0:1:M
    
    % a. evaluate
    V = par.u(C1,par) + par.beta * par.u(M-C1,par);

    % b. save max
    if V > Vstar
        Vstar = V;
        Cstar(1) = C1;
        
        Cstar(2) = M-C1;
    end

end

% 4. print results
fprintf('Vstar = %.3f, Cstar = [%d %d]',Vstar,Cstar);
%% *Brute force *$T=3$
%%
% 1. setup
par       = struct();
par.alpha = 0.5;
par.beta  = 0.90;
par.T     = 3;
par.u     = @(x,par) x.^par.alpha;

% 2. state
M = 5;

% 3. solve
Vstar = -inf;
Cstar = nan(par.T,1);
for C2 = 0:1:M
    for C1 = 0:1:(M-C2)
        
        % a. evaluate
        V = par.u(C2,par) + par.beta * par.u(C1,par) + (par.beta^2)*par.u(M-C1-C2,par);
        
        % b. save max
        if V > Vstar
            Vstar = V;
            Cstar(1) = C2;
            Cstar(2) = C1;
            Cstar(3) = M-C1-C2;
        end
        
    end
end

% 4. print results
fprintf('Vstar = %.3f, Cstar = [%d %d %d]',Vstar,Cstar);
%% *Backwards induction *$T=3$
%%
% 1. setup
par       = struct();
par.alpha = 0.5;
par.beta  = 0.00;
par.T     = 3;
par.u     = @(x,par) x.^par.alpha;

% 2. state
M = 5;

% 3. solve
Vstar = cell(par.T,1);
Cstar = cell(par.T,1);

Vstar{par.T} = par.u(0:M, par)
Cstar{par.T} = 0:M

for t = par.T-1:-1:1    
    Vstar{t} = nan(M+1,1);
    Cstar{t} = nan(M+1,1);
    % apply algorithm 4 _conditional_ on some t
end

%% Ditlev %%
