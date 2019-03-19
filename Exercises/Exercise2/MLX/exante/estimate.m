classdef estimate
methods(Static) 
        
    function h = compute_hessian(fhandle,x0,stepsize)
        
        % a. allocate
        k = numel(x0);
        hessian = zeros(k,k);
        grad = zeros(k,1);

        % b. stepsize matrix
        stepsize = stepsize*ones(k,1);
        stepsize_mat = (ones(k,1)*stepsize')';
        ee = eye(k).*stepsize_mat;

        % c. base value
        f0 = sum(fhandle(x0),1);
        
        % d. forward step
        for i=1:k
            grad(i,1) = sum(fhandle(x0+ee(:,i)),1);
        end;

        % e. "double" forward step
        for i=1:k
        for j=1:k

            hessian(i,j) = sum(fhandle(x0+(ee(:,i)+ee(:,j))),1);
            if i ~= j
                hessian(j,i) = hessian(i,j);
            end;

        end;
        end;
        
        % f. find hessian
        grad_2_mat = (ones(k,1)*grad')';
        h = (((hessian - grad_2_mat) - grad_2_mat')+ f0)./ (stepsize_mat.*stepsize_mat');
        
    end
    function [par] = updatepar(par,parnames,parvals)
        for i = 1:numel(parnames)           
            parname = parnames{i};
            parval  = parvals(i);            
            par.(parname) = parval;            
        end;        
    end     
    function [log_lik] = log_likelihood(theta,est_par,par,data)
        
        % 1. update parameters
        par = estimate.updatepar(par,est_par,theta);
           
        % 2. solve the model
        par = model.create_grids(par);
        sol = model.solve(par);

        % 3. predict        
        FILLIN;
        
        % 4. calculate errors
        FILLIN;
        
        % 5. calculate log-likelihood
        FILLIN;
    
    end
    function [] = maximum_likelihood(par,est_par,theta0,data,do_stderr)
        
        assert(numel(est_par)==numel(theta0),...
            'number of parameters and initial values do not match');
        
        % 1. estimation
        options = optimoptions('fminunc',...
                                'Display','none',...
                                'Algorithm','quasi-newton',...
                                'StepTolerance',1e-6,...
                                'FunctionTolerance',1e-6,...
                                'MaxIterations',500);
        obj_fun = @(theta) -estimate.log_likelihood(theta,est_par,par,data);
        [theta_hat,log_lik] = fminunc(FILLIN);

        % 2. standard errors (currently uncommented)
        %if do_stderr == 1
        %    cov_theta    = inv(estimate.compute_hessian(obj_fun,theta_hat,1e-5));
        %    se_theta_hat = sqrt(diag(cov_theta));
        %end
        
        % 3. print results
        fprintf('%8s = %5.3f \n','Log-lik.',-log_lik);
        for i = 1:numel(est_par)
            fprintf('%8s = %5.3f',est_par{i},theta_hat(i));
            if do_stderr == 1
                fprintf('(std. %5.3f)\n',se_theta_hat(i));
            else
                fprintf('\n')
            end
        end

    end      
    function moments = calc_moments(par,data)

        agegrid = (par.moments_minage:par.moments_maxage)-par.age_min+1;   
        moments = nanmean(data.A(:,agegrid))';
        
    end
    function obj = sum_squared_diff_moments(theta,est_par,par,data)

        % 1. update parameters
        par = estimate.updatepar(par,est_par,theta);
           
        % 2. solve the model
        par = model.create_grids(par);       
        sol = model.solve(par);   

        % 3. simulate moments
        moments = NaN(size(data.moments,1),par.moments_numsim);
        for s = 1:par.moments_numsim            
            FILLIN;
        end

        % 4. mean of moments
        FILLIN;
        
        % 5. objective function    
        FILLIN;
        
    end
    function [] = method_simulated_moments(par,est_par,theta0,data)
        
        assert(numel(est_par)==numel(theta0),...
            'number of parameters and initial values do not match');
        
        % 1. calculate data moments
        data.moments = estimate.calc_moments(par,data);
        
        % 2. estimation
        options = optimoptions('fminunc',...
                                'Display','none',...
                                'Algorithm','quasi-newton',...
                                'StepTolerance',1e-6,...
                                'FunctionTolerance',1e-6,...
                                'MaxIterations',500);
        obj_fun = @(theta) estimate.sum_squared_diff_moments(theta,est_par,par,data);
        [theta_hat,obj] = fminunc(FILLIN);
        
        % 3. print results
        fprintf('%8s = %5.3f \n','Obj.',obj);
        for i = 1:numel(est_par)
            fprintf('%8s = %5.3f\n',est_par{i},theta_hat(i));
        end
        
    end

end
end