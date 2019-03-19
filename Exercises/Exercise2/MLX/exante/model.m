classdef model   
methods(Static)
    
    function par = setup()
        
        par = struct();
        
        % 1. demograhpics
        par.T = 200;
        par.TR = par.T; % retirement age, no retirement if TR = T
        par.age_min = 25; % only relevant for figures
        
        % 2. preferences
        par.rho = 2;
        par.beta = 0.96;
        
        % 3. income parameters
        
            % growth
            par.G = 1.03;
            
            % standard deviations
            par.sigma_xi = 0.1;
            par.sigma_psi = 0.1;
        
            % low income shock
            par.low_p = 0.005; % called pi in slides
            par.low_val = 0; % called mu in slides
            
            % life-cycle
            par.L = ones(par.T,1); % if ones then no life-cycle           
        
        % 4. saving and borrowing
        par.R = 1.04;
        par.lambda = 0.0;
        
        % 5. numerical integration and grids         
        par.a_max = 20.0; % maximum point i grid for a
        par.a_phi = 1.1; % curvature parameters
        
            % number of elements
            par.Nxi  = 8; % number of quadrature points for xi
            par.Npsi = 8; % number of quadrature points for psi
            par.Na = 500; % number of points in grid for a
        
        % 6. simulation
        par.sim_mini = 2.5; % initial m in simulation
        par.simN = 500000; % number of persons in simulation
        par.simT = 100; % number of periods in simulation
        par.simlifecycle = 0; % = 0 simulate infinite horizon model
                    
    end
    function par = create_grids(par)
        
        % 1. check parameters
        assert(par.rho >= 0, 'not rho > 0');
        assert(par.lambda >= 0, 'no lambda >= 0');
            
            % is it a perfect foresight or buffer-stock model
            if par.sigma_xi == 0 && par.sigma_psi == 0 && par.low_p == 0
                par.model = 'pf'; % perfect foresight
            else
                par.model = 'bs'; % buffer-stock
            end
        
        % 2. shocks
        
            % a. basic GuassHermite
            [par.psi, par.psi_w] = funs.GaussHermite_lognorm(par.sigma_psi,par.Npsi);
            [par.xi, par.xi_w] = funs.GaussHermite_lognorm(par.sigma_xi,par.Nxi);
            
            % b. add low income shock to xi
            if par.low_p > 0
                par.xi = [par.low_val; (par.xi-par.low_p*par.low_val)/(1-par.low_p)];
                par.xi_w = [par.low_p; (1.0-par.low_p)*par.xi_w];
            end
        
            % c. vectorize all
            [par.psi_vec,par.xi_vec] = ndgrid(par.psi,par.xi); % ndgrid gives tensor product
            par.psi_vec = par.psi_vec(:);
            par.xi_vec = par.xi_vec(:);

            [par.psi_w_vec,par.xi_w_vec] = ndgrid(par.psi_w,par.xi_w); % ndgrid gives tensor product
            par.psi_w_vec = par.psi_w_vec(:);
            par.xi_w_vec = par.xi_w_vec(:);
        
            % d. vectorized total weight
            par.w = par.psi_w_vec .* par.xi_w_vec;
            assert(1-sum(par.w) < 1e-8); % == summing to 1
            
            % e. count number of shock nodes
            par.Nshocks = numel(par.w);
        
        % 3. minimum a
        if par.lambda == 0
            
            par.a_min = zeros(par.T,1); % never any borriwng
            
        else
            
            % using formula from slides 
            psi_min = min(par.psi);
            xi_min = min(par.xi);
            par.a_min = nan(par.T,1);
            for t = par.T-1:-1:1
                if t >= par.TR
                    Omega = 0; % no debt in final period
                elseif t == par.T-1
                    Omega = par.R^(-1)*par.G*par.L(t+1)*psi_min*xi_min;
                else
                    Omega = par.R^(-1)*(min(Omega,par.lambda)+xi_min)*par.G*par.L(t+1)*psi_min;
                end
                par.a_min(t) = -min(Omega,par.lambda)*par.G*par.L(t+1)*psi_min;
            end
            
        end
        
        % 4. end-of-period assets
        par.grid_a = cell(par.T,1);
        for t = 1:par.T
            par.grid_a{t} = funs.nonlinspace(par.a_min(t)+1e-6,par.a_max,par.Na,par.a_phi);
        end
        
        % 5. conditions
        par.FHW = par.G/par.R;
        par.AI = (par.R*par.beta)^(1/par.rho);
        par.GI = par.AI*sum(par.w.*par.psi_vec.^(-1))/par.G;
        par.RI = par.AI/par.R;        
        par.WRI = par.low_p^(1/par.rho)*par.AI/par.R;
        par.FVA = par.beta*sum(par.w.*(par.G*par.psi_vec).^(1-par.rho));

        % 6. set seed
        rng(2017);
               
    end
    function [] = print_and_check_parameters(par)

        fprintf('FHW = %.3f, AI = %.3f, GI = %.3f, RI = %.3f, WRI = %.3f, FVA = %.3f\n',...
           par.FHW,par.AI,par.GI,par.RI,par.WRI,par.FVA);      
        
        check for existance of solution
        if strcmp(par.model,'pf') && par.GI >= 1 && par.RI >= 1
            error('GI >= 1 && RI >= 1: no solution');
        end           
        if strcmp(par.model,'bs') && par.FVA >= 1 || par.WRI >= 1
            error('FVA >= 1 or WRI >= 1: no solution');
        end

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
    function sol = EGM_crazy_loop(sol,t,c_plus_interp,par)
        
        sol.m{t} = nan(par.Na,1);
        sol.c{t} = nan(par.Na,1);            
        for i_a = 1:par.Na
            
            avg_marg_u_plus = 0.0;
            for i_xi=1:numel(par.xi)
                xi = par.xi(i_xi);
                for i_psi = 1:numel(par.psi)
                    psi = par.psi(i_psi);
                    
                    % a. prep
                    a = par.grid_a{t}(i_a);
                    if t+1 <= par.TR
                        w = par.xi_w(i_xi)*par.psi_w(i_psi);
                        
                    else
                        w = 1/(numel(par.xi)*numel(par.psi));
                        xi = 1;
                        psi = 1;
                    end
                    
                    fac = par.G*par.L(t)*psi;
                    
                    % b. future m and c (vectors)
                    m_plus = par.R*a/fac + xi;
                    c_plus = c_plus_interp(m_plus);
                    
                    % c. average future marginal utility (number)
                    marg_u_plus = model.marg_utility(fac.*c_plus,par);
                    avg_marg_u_plus = avg_marg_u_plus + w*marg_u_plus;
                    
                end
            end
            
            % d. current c
            sol.c{t}(i_a) = model.inv_marg_utility(par.beta*par.R*avg_marg_u_plus,par);

            % e. current m
            sol.m{t}(i_a) = a + sol.c{t}(i_a);

        end  
        
    end
    function sol = EGM_loop(sol,t,c_plus_interp,par)
        
        sol.m{t} = nan(par.Na,1);
        sol.c{t} = nan(par.Na,1);            
        for i_a = 1:par.Na

            % a. prep
            a = par.grid_a{t}(i_a);
            if t+1 <= par.TR
                fac = par.G*par.L(t)*par.psi_vec;
                w = par.w;
                xi = par.xi_vec;
            else
                fac = par.G*par.L(t);
                w = 1;
                xi = 1;
            end
            inv_fac = 1.0./fac;

            % b. future m and c (vectors)
            m_plus = inv_fac*par.R*a + xi;
            c_plus = c_plus_interp(m_plus);

            % c. average future marginal utility (number)
            marg_u_plus = model.marg_utility(fac.*c_plus,par);
            avg_marg_u_plus = sum(w.*marg_u_plus,1);
            
            % d. current c
            sol.c{t}(i_a) = model.inv_marg_utility(par.beta*par.R*avg_marg_u_plus,par);

            % e. current m
            sol.m{t}(i_a) = a + sol.c{t}(i_a);

        end  
        
    end
    function sol = EGM_vec(sol,t,c_plus_interp,par)
        
        % a. prep
        a_mat = repmat(par.grid_a{t}',[par.Nshocks 1]);                             
        if t+1 <= par.TR
            fac = par.G*par.L(t)* repmat(par.psi_vec,[1 par.Na]);
            w = repmat(par.w,[1 par.Na]);
            a = a_mat;
            xi = repmat(par.xi_vec,[1 par.Na]);
        else
            fac = par.G*par.L(t)*ones(par.Na,1)';
            w = ones(par.Na,1)';
            a = par.grid_a{t}';
            xi = ones(par.Na,1)';
        end
        inv_fac = 1.0./fac;

        % b. future m and c (vectors)
        m_plus = inv_fac.*par.R.*a + xi;            
        c_plus = reshape(c_plus_interp(m_plus(:)),size(m_plus));

        % c. average future marginal utility (number)
        marg_u_plus = model.marg_utility(fac.*c_plus,par);
        avg_marg_u_plus = sum(w.*marg_u_plus,1);

        % d. current c
        sol.c{t} = model.inv_marg_utility(par.beta*par.R*avg_marg_u_plus,par)';

        % e. current m
        sol.m{t} = par.grid_a{t} + sol.c{t};
    
    end    
    function sol = EGM(sol,t,c_plus_interp,par)
        %sol = model.EGM_loop(sol,t,c_plus_interp,par);            
        sol = model.EGM_vec(sol,t,c_plus_interp,par); 
        %sol_crazy = model.EGM_crazy_loop(sol,t,c_plus_interp,par); 
    end
    function sol = solve(par)
        
        % 1. allocation solution struct and cells
        sol = struct();
        sol.m = cell(par.T,1);
        sol.c = cell(par.T,1);
        
        % 2. last period (= consume all)
        sol.m{par.T} = linspace(0,par.a_max,par.Na);
        sol.c{par.T} = sol.m{par.T};
        
        % 2. before last period
        for t = par.T-1:-1:1
            
            % a. interpolant
            c_plus_interp = griddedInterpolant(sol.m{t+1},sol.c{t+1},'linear');
            
            % b. solve by EGM
            sol = model.EGM(sol,t,c_plus_interp,par);    

            % d. add zero consumption
            sol.m{t} = [par.a_min(t); sol.m{t}];
            sol.c{t} = [0; sol.c{t}];
            
        end % t loop
                    
    end
    function sim = simulate(par,sol)
        
        sim = struct();        
        
        % 1. allocate
        sim.m = nan(par.simN,par.simT);
        sim.c = nan(par.simN,par.simT);
        sim.a = nan(par.simN,par.simT);
        sim.p = nan(par.simN,par.simT);
        sim.y = nan(par.simN,par.simT);
        
        % 2. shocks
        shocki = randsample(par.Nshocks,par.simN*par.simT,true,par.w);
        shocki = reshape(shocki,[par.simN,par.simT]);
        sim.psi = par.psi_vec(shocki);
        sim.xi = par.xi_vec(shocki);
            
            % mean of one
            assert(all(abs(mean(sim.xi(:))-1) < 1e-4));        
            assert(all(abs(mean(sim.psi(:))-1) < 1e-4));
        
        % 3. initial values
        sim.m(:,1) = par.sim_mini; 
        sim.p(:,1) = 0.0; 
        
        % 4. simulation
        for t = 1:par.simT
            
            if par.simlifecycle == 0
                c_interp = griddedInterpolant(sol.m{1},sol.c{1},'linear');
            else
                c_interp = griddedInterpolant(sol.m{t},sol.c{t},'linear');                
            end
            sim.c(:,t) = c_interp(sim.m(:,t));
            sim.a(:,t) = sim.m(:,t) - sim.c(:,t);
            
            if t < par.simT
                if t+1 > par.TR
                    sim.m(:,t+1) = par.R*sim.a(:,t) ./ (par.G*par.L(t)) +  1;
                    sim.p(:,t+1) = log(par.G) + log(par.L(t)) + sim.p(:,t);
                    sim.y(:,t+1) = sim.p(:,t+1);
                else
                    sim.m(:,t+1) = par.R*sim.a(:,t) ./ (par.G*par.L(t)*sim.psi(:,t+1)) + sim.xi(:,t+1);
                    sim.p(:,t+1) = log(par.G) + log(par.L(t)) + sim.p(:,t) + log(sim.psi(:,t+1));   
                    sim.y(:,t+1) = sim.p(:,t+1) + log(sim.xi(:,t+1));
                end
            end
                        
        end
        
        % 5. renomarlized
        sim.P = exp(sim.p);
        sim.Y = exp(sim.y);        
        sim.M = sim.m.*sim.P;
        sim.C = sim.c.*sim.P;
        sim.A = sim.a.*sim.P;
        
    end

end   
end