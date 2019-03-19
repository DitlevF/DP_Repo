function [V,Cstar,par] = vfi_finite_taseshocks(par)
    
    % 1. allocate memory
    V     = cell(par.T,2);
    Cstar = cell(par.T,2);
    if par.rho < 1.0
        par.grid_M = nonlinspace(0,par.M_max,par.NM,1.1); 
    else
        par.grid_M = nonlinspace(1e-4,par.M_max,par.NM,1.1); 
    end
    
    % 2. last period
    for L = [0,1]
        [V{par.T,L+1},Cstar{par.T,L+1}] = find_V_tasteshocks(par,L,1);             
    end
         
    % 3. backwards over time
    par.V_plus_interp = cell(2,1);
    for t = par.T-1:-1:1
        
        % a. interpolant
        par.V_plus_interp{1} = griddedInterpolant(par.grid_M,V{t+1,1},'linear');   
        par.V_plus_interp{2} = griddedInterpolant(par.grid_M,V{t+1,2},'linear');
        
        % b. find V for all discrete choices and states
        for L = [0,1]
            [V{t,L+1},Cstar{t,L+1}] = find_V_tasteshocks(par,L,0);             
        end
        
    end
        
end

function [V,Cstar] = find_V_tasteshocks(par,L,last)
    
    % loop over states    
    V     = nan(par.NM,1);
    Cstar = nan(par.NM,1);    
    for i_M = 1:numel(par.grid_M)
        
        Mt = par.grid_M(i_M);   
                
        % a. initial guess
        if i_M == 1
            initial_guess = Mt/2;
        else
            initial_guess = Cstar(i_M-1);
        end
        
        % b. find optimum
        Vfunc_neg = @(C) -value_of_choice(C,L,Mt,last,par);
        [x,fval] = fmincon(Vfunc_neg,initial_guess,[],[],[],[],...
                           0,Mt,[],par.options);                     
        
        % c. save optimum       
        V(i_M)     = -fval;
        Cstar(i_M) = x;
    
    end  
        
end

function [V] = value_of_choice(C,L,Mt,last,par)

    if last == 1 % last period
        V = par.u(C,par)-par.lambda*L;
    else
        M_plus = par.R*(Mt-C)+L;
        V1 = par.V_plus_interp{1}(M_plus);
        V2 = par.V_plus_interp{2}(M_plus);
        V  = par.u(C,par) - par.lambda*L + par.beta*logsum(V1,V2,par.sigma_epsilon);
    end
    
end