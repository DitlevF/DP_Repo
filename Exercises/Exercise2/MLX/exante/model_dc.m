classdef model_dc
methods(Static)

    function par = setup()
        
        par = struct();
        
        % 1. demograhpics
        par.T = 20;
        
        % 2. preferences
        par.rho = 2;
        par.beta = 0.96;
        par.alpha = 0.75;
        
        % 3. taste shocks
        par.sigma_eta = 0.0;
        
        % 3. income 
        par.W = 1;
        par.sigma_xi = 0.0;       
        
        % 4. saving
        par.R = 1.04;
        
        % 5. grids and numerical integration
        par.a_max = 10.0; 
        par.a_phi = 1.1; % curvature parameters
        
            % number of elements
            par.Nxi  = 1;
            par.Na = 150;
            
    end
    function par = create_grids(par)
        
        % 1. check parameters
        assert(par.rho >= 0, 'not rho > 0');
                
        % 2. shocks        
        [par.xi, par.xi_w] = funs.GaussHermite_lognorm(par.sigma_xi,par.Nxi);
                     
        % 3. end-of-period assets
        par.grid_a = cell(par.T,1);
        for t = 1:par.T
            par.grid_a{t} = funs.nonlinspace(0+1e-6,par.a_max,par.Na,par.a_phi);
        end
        
        % 4. set seed
        rng(2017);
               
    end    
    function u = utility(c,par)
        u = c.^(1-par.rho)/(1-par.rho);            
    end    
    function u = marg_utility(c,par)
        u = c.^(-par.rho);            
    end
    function c = inv_marg_utility(u,par)
        c = u.^(-1/par.rho);            
    end    
    function [v_plus,prob] = v_plus(m_plus,v_plus_interp,par)
        
        % 1. calculate v_plus
        v_plus_l = cell(2,1);        
        for l = [1,2]
            v_plus_l{l} = reshape(v_plus_interp{l}(m_plus(:)),size(m_plus));
        end
        
        % 2. find logsum and choice probabilities
        [v_plus, prob] = funs.logsum(v_plus_l{1}(:),v_plus_l{2}(:),par.sigma_eta);   
        
            % put back into shape
            v_plus = reshape(v_plus,size(m_plus));
            prob = reshape(prob(:,1),size(m_plus));
        
    end  
    function [v] = value_of_choice(m,c,v_plus_interp,par)
        
        xi_w_mat = repmat(par.xi_w,[1 numel(c)]);
        
        % 1. next-period resources
        a = m - c;   
        m_plus = par.R.*a + par.W*par.xi;            
                    
        % 2. next-period value
        v_plus_vec = model_dc.v_plus(m_plus,v_plus_interp,par);
        v_plus = sum(xi_w_mat.*v_plus_vec,1);
                
        % 3. value-of-choice
        v = model_dc.utility(c,par) + par.beta*v_plus;
        
        % 5. -infinity if choice is illegal
        I = c <= 0 | a < 0;
        v(I) = -inf;
        
    end
    function sol = EGM(sol,z_plus,t,v_plus_interp,c_plus_interp,par)
                
        % a. prep                      
        if z_plus == 2
            w = ones(par.Na,1)';
            a = par.grid_a{t}';
            xi = zeros(par.Na,1)';       
        else
            w = repmat(par.xi_w,[1 par.Na]);
            a = repmat(par.grid_a{t}',[par.Nxi 1]);   
            xi = repmat(par.xi,[1 par.Na]);    
        end

        % b. next-period ressources nad value
        m_plus = par.R.*a + par.W*xi;
        if z_plus == 2
            sol.v_plus_raw{t,z_plus} = reshape(v_plus_interp{2}(m_plus(:)),size(m_plus))';
        else
            [v_plus_vec_raw, prob] = model_dc.v_plus(m_plus,v_plus_interp,par);
            sol.v_plus_raw{t,z_plus} = sum(w.*v_plus_vec_raw,1)';
        end

        % c. next-period consumption
        if z_plus == 2
            c_plus = reshape(c_plus_interp{2}(m_plus(:)),size(m_plus));
        else
            c_plus_l = cell(2,1);  
            for j = [1,2]
                c_plus_l{j} = reshape(c_plus_interp{j}(m_plus(:)),size(m_plus));
            end
        end
        
        % d. average future marginal utility
        if z_plus == 2
            marg_u_plus = model_dc.marg_utility(c_plus,par);
        else
            marg_u_plus_l = cell(2,1);  
            for j = [1,2]
                marg_u_plus_l{j} = model_dc.marg_utility(c_plus_l{j},par);
            end
            marg_u_plus = prob.*marg_u_plus_l{1} + (1-prob).*marg_u_plus_l{2};            
        end        
        sol.avg_marg_u_plus{t,z_plus} = sum(w.*marg_u_plus,1);
        
        % e. raw c, m, and v
        sol.c_raw{t,z_plus} = model_dc.inv_marg_utility(par.beta*par.R*sol.avg_marg_u_plus{t,z_plus},par)';        
        sol.m_raw{t,z_plus} = par.grid_a{t} + sol.c_raw{t,z_plus};
        sol.v_raw{t,z_plus} = model_dc.utility(sol.c_raw{t,z_plus},par) + par.beta * sol.v_plus_raw{t,z_plus};
                
        % f. first guess
        [sol.m{t,z_plus}, I] = sort(sol.m_raw{t,z_plus});
        sol.c{t,z_plus} = sol.c_raw{t,z_plus}(I);               
        sol.v{t,z_plus} = sol.v_raw{t,z_plus}(I); 

        % g. upper envelope     
        if z_plus == 2
            return
        end

        for i = 1:numel(sol.m_raw{t,z_plus})-1
            
            FILLIN;
                    
        end % raw

    end        
    function sol = solve(par)
        
        % 1. allocation solution struct
        sol = struct();
        sol.m = cell(par.T,2);
        sol.c = cell(par.T,2);
        sol.v = cell(par.T,2);
        sol.v_plus = cell(par.T,2);
        
        sol.m_raw = cell(par.T,2);
        sol.c_raw = cell(par.T,2);
        sol.v_raw = cell(par.T,2);
        sol.v_plus_raw = cell(par.T,2);

        sol.avg_marg_u_plus = cell(par.T,2);
        
        % 2. last period (= consume all)
        for z_plus = [1,2]
            sol.m{par.T,z_plus} = linspace(0,par.a_max,par.Na);
            sol.c{par.T,z_plus} = linspace(0,par.a_max,par.Na);
            sol.v{par.T,z_plus} = model_dc.utility(sol.c{par.T,z_plus},par);
            if z_plus == 1
                sol.v{par.T,z_plus} = sol.v{par.T,z_plus} - par.alpha;
            end
        end
        
        % 3. before last period
        c_plus_interp = cell(2,1);
        for t = par.T-1:-1:1
            
            % a. create interpolants
            for z = [1,2]                
                c_plus_interp{z} = griddedInterpolant(sol.m{t+1,z},sol.c{t+1,z},'linear');
                v_plus_interp{z} = griddedInterpolant(sol.m{t+1,z},sol.v{t+1,z},'linear');                
            end
            
            % b. choice specific value functions
            for z_plus = [1,2]
                
                % i. EGM
                sol = model_dc.EGM(sol,z_plus,t,v_plus_interp,c_plus_interp,par);                       
                        
                % ii. add 10 points on consstraint
                if z_plus == 1
                    
                    c_con = linspace(0+1e-6, sol.c{t,z_plus}(1)-1e-6,10);
                    m_con = c_con;
                    v_con = model_dc.value_of_choice(m_con,c_con,v_plus_interp,par);
                
                    sol.m{t,z_plus} = [m_con'; sol.m{t,z_plus}];
                    sol.c{t,z_plus} = [c_con'; sol.c{t,z_plus}];
                    sol.v{t,z_plus} = [v_con'; sol.v{t,z_plus}];
                    
                end
                
                % ii. deduct disutility of labor
                if z_plus == 1                
                    sol.v{t,z_plus} = sol.v{t,z_plus} - par.alpha;
                    sol.v_raw{t,z_plus} = sol.v_raw{t,z_plus} - par.alpha;
                end

            end
            
        end % t loop
                    
    end  
    
end  
end