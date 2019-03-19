classdef figs_dc
methods(Static)
    
    function [] = consumption_function_convergence(par,sol,ts,z_plus)
        
        fig = figure('Name',sprintf('converge_c_zplus%d_t%d_t%d_%s',...
            z_plus,ts(1),ts(end),par.prefix));
        hold('on') ;
        
        for t = ts
            h = plot(sol.m{t,z_plus},sol.c{t,z_plus},...
                 'o','MarkerSize',3,'DisplayName',sprintf('$t = %d$',t));
            set(h, 'MarkerFaceColor', get(h, 'Color'));
        end
        
        % limits
        xlim([0, 5])
        ylim([0, 3])
        
        % layout
        xlabel('$m_t$');
        ylabel(sprintf('$c(m_t,z_{t+1} = %d)$',z_plus-1));
        legend('Location','best');
        box('on');
        grid on;
        
        funs.printfig(fig);
    
    end
    function [] = value_function_convergence(par,sol,ts,z_plus)
        
        fig = figure('Name',sprintf('converge_v_zplus%d_t%d_t%d_%s',...
            z_plus,ts(1),ts(end),par.prefix));
        hold('on')        
        
        for t = ts
            h = plot(sol.m{t,z_plus},sol.v{t,z_plus},...
                 'o','MarkerSize',3,'DisplayName',sprintf('$t = %d$',t));
            set(h, 'MarkerFaceColor', get(h, 'Color'));
        end
        
        % limits
        xlim([0, 5])
        ylim([-20, 0])
        
        % layout
        xlabel('$m_t$');
        ylabel(sprintf('$v(m_t,z_{t+1} = %d)$',z_plus-1));
        legend('Location','best');
        box('on');
        grid on;
        
        funs.printfig(fig);
    
    end    
    function [] = choice_specific_value_functions(par,sol,t)
        
        fig = figure('Name',sprintf('v_zplus_t%d_%s',t,par.prefix));
        
        plot(sol.m{t,1},sol.v{t,1},'-o',...
            'linewidth',1.5,'MarkerSize',3,'color','black',...
            'DisplayName',sprintf('$v_{%d}(m_t,z_{t+1} = 0)$',t))
        hold on;
        plot(sol.m{t,2},sol.v{t,2},'-o',...
            'linewidth',1.5,'MarkerSize',3,'color','red',...
            'DisplayName',sprintf('$v_{%d}(m_t,z_{t+1} = 1)$',t))

        % limits
        xlim([0 5])
        ylim([-5 0])
        
        % layout
        xlabel('$m_t$')
        ylabel('$v$')
        legend('Location','best');
        box('on');
        grid on;
        
        funs.printfig(fig);
        
    end
    function [] = value_of_choice(par,sol,t)    
        
        % 1. total
        fig = figure('Name',sprintf('val_choice_%s_t%d',par.prefix,t));
        
        hold on;
        for m = [1.7 1.8 1.9 2.0]
            a = par.grid_a{t};
            c = m - a;
            I = c > 0;
            v = model_dc.utility(c(I),par) + par.beta*sol.v_plus_raw{t,1}(I);

            plot(c(I),v,'-o',...
                'linewidth',1.5,'MarkerSize',3,...
                'DisplayName',sprintf('$m_t = %.1f $',m))
        
        end
        
        % limits
        xlim([0.5 2])
        ylim([-3.2 -2.7])
        
        % layout
        xlabel('$c_t$')
        ylabel('value-of-choice')
        legend('Location','best');
        box('on');
        grid on;
        
        funs.printfig(fig);
        
        % 2. utility
        fig = figure('Name',sprintf('val_choice_util_%s_t%d',par.prefix,t));
        
        hold on;
        for m = [1.7 1.8 1.9 2.0]
            a = par.grid_a{t};
            c = m - a;
            I = c > 0;
            v = model_dc.utility(c(I),par);

            plot(c(I),v,'-o',...
                'linewidth',1.5,'MarkerSize',3,...
                'DisplayName',sprintf('$m_t = %.1f $',m))
        
        end
        
        % limits
        xlim([0.5 2])
        
        % layout
        xlabel('$c_t$')
        ylabel('value-of-choice')
        legend('Location','best');
        box('on');
        grid on;
        
        funs.printfig(fig);
        
        % 3. continuation value
        fig = figure('Name',sprintf('val_choice_contval_%s_t%d',par.prefix,t));
        
        hold on;
        for m = [1.7 1.8 1.9 2.0]
            a = par.grid_a{t};
            c = m - a;
            I = c > 0;
            v = par.beta*sol.v_plus_raw{t,1}(I);

            plot(c(I),v,'-o',...
                'linewidth',1.5,'MarkerSize',3,...
                'DisplayName',sprintf('$m_t = %.1f $',m))
        
        end
        
        % limits
        xlim([0.5 2])
        
        % layout
        xlabel('$c_t$')
        ylabel('value-of-choice')
        legend('Location','best');
        box('on');
        grid on;
        
        funs.printfig(fig);        
        
    end
    function [] = EGM(par,sol,t)
        
        fig = figure('Name',sprintf('EGM_%s_t%d',par.prefix,t));
        
        plot(par.grid_a{t},par.beta*par.R*sol.avg_marg_u_plus{t,1},'-o',...
            'linewidth',1.5,'MarkerSize',3,'color','black',...
            'DisplayName','$\beta R E[u^{\prime}(c_{t+1}(Ra_t+W\xi_{t+1},z_{t+2}))]$')
        hold on;

        plot(par.grid_a{t},sol.c_raw{t,1},'-o',...
            'linewidth',1,'color','red',...
            'MarkerSize',4,'MarkerFaceColor','red',...
            'DisplayName','$c_t = \beta R E[u^{\prime}(c_{t+1}(Ra_t+W\xi_{t+1},z_{t+2}))]^{-1}$')

        plot(par.grid_a{t},sol.m_raw{t,1},'-o',...
            'linewidth',1,'color','blue',...
            'MarkerSize',4,'MarkerFaceColor','blue',...
            'DisplayName','$m_t = a_t + c_t$')

        % limits
        xlim([0 2])
        
        % layout
        xlabel('$a_t$')
        ylabel('')
        legend('Location','best');
        box('on');
        grid on;
        
        funs.printfig(fig);
        
        % add numbering
        m_low = 1.6;
        m_high = 2.3;        
        I = sol.m_raw{t,1} >= m_low & sol.m_raw{t,1} <= m_high;
        m = sol.m_raw{t,1}(I);
        c = sol.c_raw{t,1}(I);     
        a = par.grid_a{t}(I);     
        for i = 1:numel(m)
            text(a(i),c(i),sprintf('%d',i),'color','blue',...
            'VerticalAlignment','bottom','HorizontalAlignment','right');  
            text(a(i),m(i),sprintf('%d',i),'color','blue',...
            'VerticalAlignment','bottom','HorizontalAlignment','right');          
        end
        set(fig,'name',sprintf('EGM_numbering_%s_t%d',par.prefix,t),...
            'numbertitle','off')
        legend('Location','best'); 
        
        funs.printfig(fig);        
        
    end   
    function [] = upperenvelope(par,sol,t)
        
        m_low = 1.6;
        m_high = 2.3;
        
        %%%%%
        % c %
        %%%%%
        
        fig = figure('Name',sprintf('upperenvelope_c_without_%s_t%d',par.prefix,t));
        
        % raw
        plot(sol.m_raw{t,1},sol.c_raw{t,1},'-o',...
            'linewidth',1,'color','black',...
            'MarkerSize',4,'MarkerFaceColor','black',...
            'DisplayName','raw $c(m_t)$')
        hold on;

        % numbering and limits
        I = sol.m_raw{t,1} >= m_low & sol.m_raw{t,1} <= m_high;
        m = sol.m_raw{t,1}(I);
        c = sol.c_raw{t,1}(I);
        for i = 1:numel(m)
            text(m(i),c(i),sprintf('%d',i),'color','blue',...
            'VerticalAlignment','bottom','HorizontalAlignment','right');  
        end
                
        % limits
        xlim([m_low m_high])
                
        % layout
        xlabel('$m_t$')
        ylabel('$c(m_t,z_{t+1})$')
        legend('Location','best');
        grid on;

        funs.printfig(fig);

        % optimal
        h = scatter(sol.m{t,1},sol.c{t,1},80,'red','s','filled',...
            'DisplayName','optimal $c(m_t)$');
        alpha(h,'0.75');        
        uistack(h,'bottom');
        
        set(fig,'name',sprintf('upperenvelope_c_%s_t%d',par.prefix,t),...
            'numbertitle','off')
        legend('Location','best');        
        funs.printfig(fig);
        
        
        %%%%%
        % v %
        %%%%%
        
        fig = figure('Name',sprintf('upperenvelope_v_without_%s_t%d',par.prefix,t));

        % raw
        plot(sol.m_raw{t,1},sol.v_raw{t,1},'-o',...
            'linewidth',1,'color','black',...
            'MarkerSize',4,'MarkerFaceColor','black',...
            'DisplayName','raw $v(m_t)$')
        hold on;

        % numbering and limits
        I = sol.m_raw{t,1} >= m_low & sol.m_raw{t,1} <= m_high;
        m = sol.m_raw{t,1}(I);
        v = sol.v_raw{t,1}(I);
        for i = 1:numel(m)
            text(m(i),v(i),sprintf('%d',i),'color','blue',...
            'VerticalAlignment','bottom','HorizontalAlignment','right');   
        end
        
        % limits
        xlim([m_low m_high])

        % layout
        xlabel('$m_t$')
        ylabel('$v(m_t,z_{t+1})$')
        legend('Location','best');
        grid on;
                
        funs.printfig(fig);
        
        % optimal
        h = scatter(sol.m{t,1},sol.v{t,1},80,'red','s','filled',...
            'DisplayName','optimal $v(m_t)$');
        alpha(h,'0.75');        
        uistack(h,'bottom');
        set(fig,'name',sprintf('upperenvelope_v_%s_t%d',par.prefix,t),...
            'numbertitle','off')                
        legend('Location','best');        
        funs.printfig(fig);       
   
    end

end
end