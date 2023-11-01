% Before running the Q-learning algorithm, solve optimal flow for
% - each hour of day (24) 
% - over the annual period (365)
% - over the choices for daily total flow allocation
% Here, year 2018 hourly prices used for hourly flow optimization,
% so refer as optimFlow_2018 the solved flows and  
% optimRevenue_2018 the daily revenue given optimized hourly flow.

% Daily revenue when choosing action index ind:
load('optimRevenue_2018.mat')
% Optimized hourly flow when choosing action index ind: 
load('optimFlow_2018.mat')

%%%%% A. STATE SPACE %%%%%
% The day-of year index over the annual inflow realization years: 
load('daily_index.mat')
number_of_days_annual = 365;
% State variable is the day-of-year
stateDayIndex = 1:annual;
% Inlow difference to weekly average percentiles
load('daily_diff_to_week_mean_ind.mat')
% State varuable is the difference of realized inflow to historical mean
stateInflowDiff = 1:max(daily_diff_to_week_mean_ind); 

load('daily_inflow.mat')
number_of_years = (length(daily_inflow)-1)/number_of_days_annual;

% Upper reservoi min and max water level (cm)
waterHeightMin = 24360; % (cm)
waterHeightMax = 24465; % (cm)
% Set e.g. 100 discrete points for water level
waterLevelPoints = 100;
% State variable is the water level of the upper reservoir
stateWaterHeight = linspace(waterHeightMin,waterHeightMax,waterLevelPoints);

% Set up the starting index of water height in each episode (simulated year)
load('average_first_day_height.mat') % (m)
average_first_day_height = average_first_day_height*100; % to (cm)
[~,init_water_level_ind] = min(abs(stateWaterHeight-average_first_day_height));

%%%%% B. ACTION SPACE %%%%%
maxTurbineFlow = 16; % (m^3/s)
minTurbineFlow = 0; % (m^3/s)
turbine_flow_optim_step_size = 0.5;
actionDispatch = minTurbineFlow:turbine_flow_optim_step_size:maxTurbineFlow;

%%%%% C. Set up the Q_table  %%%%%
Q_table = zeros(length(stateDayIndex),length(stateInflowDiff),length(stateWaterHeight),length(actionDispatch));

%%%% D. Hydropower parameters %%%%%
load('rho.mat') % rho is the efficiency as a function of flow rate
g = 9.81; % gravitation (m/s^2)
head = 10; % (m)
density = 1000; % (1000 kg/m^3)
seconds_in_hour = 3600;
hours_in_day = 24;

%%%% Market parameters %%%%
% Average up-balancing price premium for missing energy,
% used in case min. water level limit would be breached and the 
% hydropower plant must buy the missing energy at average up-balancing premium
load('averageUpPricePremium.mat')

%%%% Water level transition function parameters %%%%
% 13824 (m^3) net flow diff. equals 1 (cm) change in water level 
alpha = 1/13824; % (m^3) to (cm)
% Evaporation (cm per day):
evap = zeros(length(stateDayIndex),1);
evap(121:151,1) = 0.2592*ones(length(121:151),1); % May
evap(152:181,1) = 0.2729*ones(length(152:181),1); % June
evap(182:212,1) = 0.2607*ones(length(182:212),1); % July
evap(213:243,1) = 0.1881*ones(length(213:243),1); % Aug
evap(244:273,1) = 0.0695*ones(length(244:273),1); % Sep

% Number of training period days:
training_period_days = length(daily_inflow)-1;

%%%% E. Other parameters %%%%

% Discounting:
discount_rate = 0.05;
discount_factor = (1/(1+discount_rate));
beta = discount_factor.^(1/number_of_days_annual);
discount_factor_annual = zeros(number_of_days_annual,1);
for i = 1:number_of_days_annual
    discount_factor_annual(i) = beta.^(i);
end

discount_factor = zeros(training_period_days,1);
for year = 1:number_of_years
    discount_factor(1+(year-1)*365:365+(year-1)*365,1) = discount_factor_annual;
end

% Exploration/exploitation and learning
exploitation_parameter = 0.0005; 
learning_parameter = 500; 
% Number of learning rounds:
number_of_rounds = 10000; 

%%%% Performance check in every 100th round
% - column 1: total reward
% - column 2: share of non-visit Q-table cells in the simulation
% - column 3: share of lower water limit breaks in the simulation
check_at_learning_round = 100;
performance = zeros(number_of_rounds/check_at_learning_round,3);

%%%% F. Q-learning algorithm %%%%%
for learning_round = 1:number_of_rounds
    % First day water level:
    water_level_ind = init_water_level_ind;
    hour_init = 1;
    state = [daily_index(hour_init),daily_diff_to_week_mean_ind(hour_init),water_level_ind];
    for day = 1:training_period_days
        %%% EXPLORATION vs EXPLOITATION: %%%
        % epsilon greedy action choice: choose the best action with probability Prob
        Prob = 1 - exp(-exploitation_parameter*learning_round);
        %%% LEARNING %%%
        learning_rate = learning_parameter./(learning_parameter + learning_round - 1);
        
        if rand > Prob || sum(Q_table(state(1),state(2),state(3),:)) == 0
            action_ind = randi([1 length(actionDispatch)],1,1);
        else
           [~,action_ind] = max(Q_table(state(1),state(2),state(3),:));
        end

        % Flow dispatch (average daily flow) (m^3/s):
        flow_dispatch = actionDispatch(action_ind);
        % Revenue for daily flow index: action_ind
        revenue = optimRevenue_2018(daily_index(day),action_ind);
        % Water level: 
        net_flow = (daily_inflow(day) - flow_dispatch)*seconds_in_hour*hours_in_day; % (m^3)
        % alpha = 1/13824, i.e., 13824 (m^3) net flow diff. equals 1 (cm) in level 
        next_water_level = stateWaterHeight(state(3)) + alpha*net_flow - evap(daily_index(day)); % (cm)
        % Check if next day's water level would go above max: spilling
        if next_water_level > stateWaterHeight(end) 
            reward = revenue; 
        % Check if water level between min and max:
        elseif next_water_level >= stateWaterHeight(1)
            reward = revenue;
        % Check if next day's water level would go below min: 
        % This is not allowed. Use a large fine in training, to avoid this. 
        % (Later when using the Q-table, buy the missing energy at expected
        % up-regulating price).
        else % next_water_level < stateWaterHeight(1)
            missing_water = (next_water_level - stateWaterHeight(1))*(1/alpha); % (m^3)
            new_daily_total = flow_dispatch*(seconds_in_hour*hours_in_day) + missing_water; % (m^3)
            new_average_dispatch = new_daily_total/(seconds_in_hour*hours_in_day); % (m^3/s)
            [~,corrected_action_ind] = min(abs(actionDispatch-new_average_dispatch));
            water_level_fine = -1e6; 
            next_water_level = stateWaterHeight(1);
            reward = revenue + water_level_fine;
        end
        [~,next_water_level_ind] = min(abs(stateWaterHeight-next_water_level));
        % Next state:
        if day == training_period_days
            next_day = 1;
        else
            next_day = day + 1;
        end
        new_state = [daily_index(next_day),daily_diff_to_week_mean_ind(next_day),next_water_level_ind]; 
        % Update Q-table
        next_value = max(Q_table(new_state(1),new_state(2),new_state(3),:));
        error = (reward + beta*next_value) - Q_table(state(1),state(2),state(3),action_ind);

        Q_table(state(1),state(2),state(3),action_ind) = ...
        Q_table(state(1),state(2),state(3),action_ind) + learning_rate*error;

        state = new_state;
    end
formatSpec = 'Iteration round %4.0f: exploitation prob. %4.3f and learning rate %4.3f\n';
fprintf(formatSpec,learning_round,Prob,learning_rate)

% Check Q-table performance in every 100th round:
    if mod(learning_round,check_at_learning_round) == 0
       [zeroCells,limitBreaks,totalReward] = check_Qtable_performance(Q_table, ...
        init_water_level_ind, stateWaterHeight, actionDispatch, ...
        daily_index, daily_diff_to_week_mean_ind, alpha, evap, discount_factor, ...
        daily_inflow, optimRevenue_2018, training_period_days);  
    performance(learning_round/100,1) = sum(totalReward);
    performance(learning_round/100,2) = sum(zeroCells/training_period_days);
    performance(learning_round/100,3) = sum(limitBreaks/training_period_days);
    end
end

% Use Q-table in 2000 - 2017 inflow years (training environment):
[simul_hourly_flow_train, simul_hourly_spilling_train, ...
 daily_flow_dispatch_train, daily_flow_dispatch_correction_train, revenue_train, balancing_fine_train, water_level_train, spilling_train] = ...
use_Qtable_2000_2017(Q_table, ...
init_water_level_ind, stateWaterHeight, actionDispatch, ...
daily_index, daily_diff_to_week_mean_ind, alpha,evap, ...
daily_inflow, optimRevenue_2018, optimFlow_2018, averageUpPricePremium);

% Use Q-table in 2018 inflow year (previously unseen operation environment): 
load('day_of_year_2018.mat')
load('daily_diff_to_week_mean_ind_2018.mat')
load('inFlow_2018.mat')

[simul_hourly_flow, simul_hourly_spilling, ...
    daily_flow_dispatch, daily_flow_dispatch_correction, revenue, balancing_fine, water_level, spilling] = ...
use_Qtable_2018(Q_table, ...
init_water_level_ind, stateWaterHeight, actionDispatch, ...
day_of_year_2018, daily_diff_to_week_mean_ind_2018, alpha, evap, ...
inFlow_2018, optimRevenue_2018, optimFlow_2018, averageUpPricePremium);  