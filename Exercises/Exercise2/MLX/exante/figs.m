classdef figs
methods(Static)
    
    % solution
    function [] = consumption_function_convergence(par,sol)
        
        fig = figure('Name',sprintf('cons_converge_%s',par.prefix));
        hold('on')
        
        % consumption function for various t
        for t = [par.T, par.T-1, par.T-5, par.T-10, 101, 51, 1]
            h = plot(sol.m{t},sol.c{t},...
                 '-','linewidth',1.5,'DisplayName',sprintf('$n = %d$',par.T-t+1));
            set(h, 'MarkerFaceColor', get(h, 'Color'));
        end
        
        % limits
        xlim([min(par.a_min), 5])
        ylim([0, 5])
        
        % layout
        text(1.5,0.2,sprintf('%s: $\\beta = %3.2f$, $R = %3.2f$, $G = %3.2f$',...
            strrep(par.prefix,'_',' '),par.beta,par.R,par.G),...,
            'BackgroundColor','white','interpreter','latex');
        xlabel('$m_t$');
        ylabel('$c(m_t)$');
        legend('Location','northwest');
        box('on');
        grid on;
        
        funs.printfig(fig);
    
    end
    function [] = consumption_function_convergence_age(par,sol)
        
        fig = figure('Name',sprintf('cons_converge_%s',par.prefix));
        hold('on')
        
        % consumption function for various ages
        for age = [26, 35, 45, 55, 65, 75, par.T+par.age_min-1, par.T+par.age_min]
            h = plot(sol.m{age-par.age_min},sol.c{age-par.age_min},...
                 '-','linewidth',1.5,'DisplayName',sprintf('age $= %d$',age));
            set(h, 'MarkerFaceColor', get(h, 'Color'));
        end
        
        % limits
        xlim([min(par.a_min), 5])
        ylim([0, 5])
        
        % layout
        text(1.5,0.2,sprintf('%s: $\\beta = %3.2f$, $R = %3.2f$, $G = %3.2f$',...
            strrep(par.prefix,'_',' '),par.beta,par.R,par.G),...,
            'BackgroundColor','white','interpreter','latex');
        xlabel('$m_t$');
        ylabel('$c(m_t)$');
        legend('Location','northwest');
        box('on');
        grid on;
        
        funs.printfig(fig);
    
    end
    
    function [] = consumption_function_pf(par,sol)
        
        fig = figure('Name',sprintf('cons_converge_pf_%s',par.prefix));
        hold('on')
        
        t = 1;
        
        % perfect foresight consumption
        c_pf = (1-par.RI)*(sol.m{t}+(1-par.FHW)^(-1)-1);   

        % consumption function deviation from perfect foresight
        plot(sol.m{t},sol.c{t}-c_pf,...
            '-','linewidth',1.5);

        % limits
        xlim([1, 500]);
        ylim_now = get(gca,'ylim');
        if max(abs(ylim_now)) < 1e-4
            ylim([-1 1]);
        end
        
        % layout
        xlabel('$m_t$')
        ylabel('$c(m_t) - c^{PF}(m_t)$')
        box('on')
        grid on;
        
        funs.printfig(fig);
    
    end    
    function [] = buffer_stock_target(par,sol)
        
        % 1. find a and avg. m_plus and c_plus
        c_plus_interp = griddedInterpolant(sol.m{1},sol.c{1},'linear');    
            
            % allocate
            a = nan(par.Na+1,1);
            m_plus = nan(par.Na+1,1);
            C_plus = nan(par.Na+1,1);
            
            delta_log_C_plus = nan(par.Na+1,1);
            delta_log_C_plus_approx_2 = nan(par.Na+1,1);
        
        fac = 1.0./(par.G*par.psi_vec);
        for i_a = 1:(par.Na+1)
            
            % a. a and m
            a(i_a) = sol.m{1}(i_a)-sol.c{1}(i_a);            
            m_plus(i_a) = sum(par.w.*(fac*par.R*a(i_a) + par.xi_vec));                
            
            % b. C_plus
            m_plus_vec = fac*par.R*a(i_a) + par.xi_vec;            
            C_plus_vec = par.G*par.psi_vec.*c_plus_interp(m_plus_vec);
            C_plus(i_a) = sum(par.w.*C_plus_vec);
            
            % c. approx 
            delta_log_C_plus(i_a) = sum(par.w.*(log(par.G*C_plus_vec)))-log(sol.c{1}(i_a));
            var_C_plus = sum(par.w.*(log(par.G*C_plus_vec) - log(sol.c{1}(i_a)) - delta_log_C_plus(i_a)).^2);
            delta_log_C_plus_approx_2(i_a) = par.rho^(-1)*(log(par.R*par.beta)) + 2/par.rho*var_C_plus + log(par.G);
        
        end
        
        % 2. find target
        [~, i] = min(abs(m_plus-sol.m{1}));
        m_target = sol.m{1}(i);

        % 3. figure 1 - buffer-stock target
        fig = figure('Name',sprintf('buffer_stock_target_%s',par.prefix));        
        hold('on')
                
        h = plot(sol.m{1},sol.c{1},...
                 '-','linewidth',1.5,'DisplayName','$c(m_t)$');
        set(h, 'MarkerFaceColor', get(h, 'Color'));
                    
        h = plot(sol.m{1},a,...
                 '-','linewidth',1.5,'DisplayName','$a_t=m_t-c(m_t)$');
        set(h, 'MarkerFaceColor', get(h, 'Color'));
                    
        h = plot(sol.m{1},m_plus,'-','linewidth',1.5,...
            'DisplayName','$E[m_{t+1} | a_t]$');
        set(h, 'MarkerFaceColor', get(h, 'Color'));
                    
        % 45 degree line
        plot([0 5],[0 5],'-','linewidth',1.5,'color','black',...
            'DisplayName','45 degree');
             
        % target
        if strcmp(par.model,'bs') && par.GI < 1
            plot([m_target m_target],[0 5],'--','linewidth',1.5,'color','black',...
                'DisplayName','target');
        end 
        
        % perfect foresight solution
        if par.FHW < 1 && par.RI < 1
            c_pf = (1-par.RI)*(sol.m{1}+(1-par.FHW)^(-1)-1);   
            plot(sol.m{1},c_pf,':','linewidth',1.5,'color','black',...
                'DisplayName','$c^{PF}(m_t)$');
        end
        
        % limits
        xlim([min(par.a_min), 5])
        ylim([0, 5])

        % layout
        text(1.5,0.2,sprintf('%s: $\\beta = %3.2f$, $R = %3.2f$, $G = %3.2f$',...
            strrep(par.prefix,'_',' '),par.beta,par.R,par.G),...,
            'BackgroundColor','white','interpreter','latex');
        xlabel('$m_t$')
        ylabel('')
        legend('Location','best')
        box('on')
        grid on;
        
        funs.printfig(fig);
           
        % STOP
        if strcmp(par.model,'pf')
            return;
        end
        
        % 4. figure 2 - c ratio
        fig = figure('Name',sprintf('cons_growth_%s',par.prefix));            
        hold('on')
                
        h = plot(sol.m{1},(C_plus./sol.c{1}),'-','linewidth',1.5,...
            'DisplayName','$E[C_{t+1}/C_t]$');                
        set(h, 'MarkerFaceColor', get(h, 'Color'));
                    
        plot([m_target m_target],[0 10],'--','linewidth',1.5,'color','black',...
            'DisplayName','target');

        plot([min(par.a_min) 500],[par.G par.G],...
            ':','linewidth',1.5,'color','black',...
            'DisplayName','$G$');
        
        plot([min(par.a_min) 500],[(par.R*par.beta)^(1/par.rho) (par.R*par.beta)^(1/par.rho)],...
            '-','linewidth',1.5,'color','black',...
            'DisplayName','$(\beta R)^{1/\rho}$');
                
        % limit     
        xlim([min(par.a_min), 10]);
        ylim([0.95 1.1]);
        
        % layout
        xlabel('$m_t$')
        ylabel('$C_{t+1}/C_t$')
        legend('Location','northeast')
        box('on')
        grid on;
        
        funs.printfig(fig);
        
        % 4. figure 3 - euler approx
        fig = figure('Name',sprintf('euler_approx_%s',par.prefix));            
        hold('on')
                
        h = plot(sol.m{1},delta_log_C_plus,'-','linewidth',1.5,...
            'DisplayName','$E[\Delta \log C_{t+1}]$');                
        set(h, 'MarkerFaceColor', get(h, 'Color'));

        h = plot(sol.m{1},par.rho^(-1)*log(par.R*par.beta)*ones(par.Na+1,1)+log(par.G),'-','linewidth',1.5,...
            'DisplayName','1st order approx.');                
        set(h, 'MarkerFaceColor', get(h, 'Color'));
        
        h = plot(sol.m{1},delta_log_C_plus_approx_2,'-','linewidth',1.5,...
            'DisplayName','2nd order approx.');                
        set(h, 'MarkerFaceColor', get(h, 'Color'));
        
        plot([m_target m_target],[-10 10],'--','linewidth',1.5,'color','black',...
            'DisplayName','target');
                
        % limit     
        xlim([min(par.a_min), 10]);
        ylim([-0.03 0.12])
        
        % layout
        xlabel('$m_t$')
        ylabel('$E[\Delta \log C_{t+1}]$')
        legend('Location','best')
        box('on')
        grid on;
        
        funs.printfig(fig);
        
    end
        
    % simulation
    function [] = simulate_cdf_cash_on_hand(par,sim)
        
        % 1. consumption growth
        fig = figure('Name',sprintf('sim_cdf_cash_on_hand_%s',par.prefix));
        hold('on')

        for t = [1, 2, 3, 5, 10, 30, 50, par.simT]
           [y,x] = ecdf(sim.m(:,t));
           plot(x,y,'linewidth',1.5,'DisplayName',sprintf('$t = %d$',t));
        end
        
        % limits
        xlim([min(par.a_min), 4])
        
        % layout  
        xlabel('$m_t$')
        ylabel('CDF')
        legend('Location','best')
        box('on')
        grid on;
        
        funs.printfig(fig);           
        
    end
    function [] = simulate_consumption_growth(par,sim)
        
        % 1. consumption growth
        fig = figure('Name',sprintf('sim_cons_growth_%s',par.prefix));
        hold('on')

        h = plot(1:par.simT-1,mean(log(sim.C(:,2:end))-log(sim.C(:,1:end-1)),1),...
            '-','linewidth',1.5,...
            'DisplayName','$E[\Delta\log(C_t)]$');
        set(h, 'MarkerFaceColor', get(h, 'Color'));
                    
        h = plot(1:par.simT-1,log(mean(sim.C(:,2:end),1))-log(mean(sim.C(:,1:end-1),1)),...
            '-','linewidth',1.5,...
            'DisplayName','$\Delta\log(E[C_t])$');
        set(h, 'MarkerFaceColor', get(h, 'Color'));
                    
        hline = log(par.G);
        plot([0 par.simT],[hline hline],...
            '-','linewidth',1.5,'color','black',...
            'DisplayName','$\log(G)$');
        
        hline = log(par.G)-0.5*par.sigma_psi^2;
        plot([0 par.simT],[hline hline],...
            '--','linewidth',1.5,'color','black',...
            'DisplayName','$\log(G)-0.5\sigma_{\psi}^2$');
           
        % layout  
        xlabel('time')
        ylabel('')
        legend('Location','best')
        box('on')
        grid on;
        
        funs.printfig(fig);
       
        % 2. cash-on-hand
        fig = figure('Name',sprintf('sim_cash_on_hand_%s',par.prefix));
        hold('on')

        t = 1;
        plot(1:par.simT,mean(sim.m,1),...
            '-','linewidth',1.5);
        plot(1:par.simT,prctile(sim.m,25,1),...
            '--','linewidth',1.5,'color','black');
        plot(1:par.simT,prctile(sim.m,75,1),...
            '--','linewidth',1.5,'color','black');
        
        % layout 
        xlabel('time')
        ylabel('$E[m_t]$')
        box('on')
        grid on;  
        
        funs.printfig(fig);
        
    end
    
    % compare
    function [] = consumption_function_compare(pars,sols,prefix)
        
        fig = figure('Name',sprintf('compare_cons_func_%s',prefix));
        hold('on')
    
        t = 1;
        for i = 1:numel(pars)
            par = pars{i};
            sol = sols{i};  
            leg = sprintf('$\\beta = %5.2f, \\rho = %5.2f, R = %5.2f, G = %5.2f$',...
                par.beta,par.rho,par.R,par.G);  
            h = plot(sol.m{t},sol.c{t},...
                 '-','linewidth',1.5,...
                 'DisplayName',leg);
            set(h, 'MarkerFaceColor', get(h, 'Color'));
        end

        % limits
        xlim([min(par.a_min), 5])
        ylim([0, 2])

        % layout
        xlabel('$m_t$')
        ylabel('$c(m_t)$')
        legend('Location','northwest')
        box('on')
        grid on;
        
        funs.printfig(fig);
    
    end
    
    % life-cycle
    function [] = life_cycle_income(par,sim)
        
        fig = figure('Name',sprintf('sim_Y_%s',par.prefix));
        hold('on')

        plot(par.age_min+(1:par.simT),mean(sim.Y,1),...
            '-','linewidth',1.5);
        
        % layout 
        ylabel('income, $Y_t$')
        xlabel('age')
        box('on')
        grid on;  
        
        funs.printfig(fig);
        
    end
    function [] = life_cycle_cashonhand(par,sim)
        
        fig = figure('Name',sprintf('sim_M_%s',par.prefix));
        hold('on')

        plot(par.age_min+(1:par.simT),mean(sim.M,1),...
            '-','linewidth',1.5);
        
        % layout 
        ylabel('cash-on-hand, $M_t$')        
        xlabel('age')
        box('on')
        grid on;  
        
        funs.printfig(fig);
        
    end    
    function [] = life_cycle_consumption(par,sim)
        
        fig = figure('Name',sprintf('sim_C_%s',par.prefix));
        hold('on')

        plot(par.age_min+(1:par.simT),mean(sim.C,1),...
            '-','linewidth',1.5);

        % layout 
        ylabel('consumption, $C_t$')         
        xlabel('age')
        box('on')
        grid on;  
        
        funs.printfig(fig);
        
    end       
    function [] = life_cycle_assets(par,sim)
        
        fig = figure('Name',sprintf('sim_A_%s',par.prefix));
        hold('on')

        plot(par.age_min+(1:par.simT),mean(sim.A,1),...
            '-','linewidth',1.5);

        % layout 
        ylabel('assets, $A_t$')         
        xlabel('age')
        box('on')
        grid on;  
        
        funs.printfig(fig);
        
    end    
    
end
end
