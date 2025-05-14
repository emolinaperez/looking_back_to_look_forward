#set root directory
 root <- "/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/63. Looking Back to Look Forward/looking_back_to_look_forward/"
#source model structure  
 model_version <- "model/bardis_model.r"
 source(paste0(root,model_version))

#set initial conditions for state variables 
 state <- c(Resources = 1.0, 
            Economy = 1.0, 
            Bureaucracy = 1.0, 
            Pollution = 1.0)

#set dynamic equlibrium parameter vector 
p_0 <- c(
  k_resources =  0.15*0.5 , # Autoregeneration rate of resources  
  ef_economy_resources_on_prod = .08*1.5,          # Production rate
  ef_bureaucracy_on_prod = 0.02*1.5, # Effect of bureaucracy on production 
  k_deprec = 0.01,        # Depreciation rate
  ef_pollution_on_depreciation = 0.05, # Effect of pollution on economy depreciation 
  k_bureaucracy = 0.01,    # Bureaucracy formation rate
  ef_economy_on_bureaucracy = 0.03, # Effect of the Economy of bureaucracy formation 
  k_decay_bureaucracy = 0.02,          # Bureaucracy decay rate
  ef_pollution_on_bureaucracy = 0.02, # Effect of pollution on bureaucracy decay  
  k_pollution = 0.05, #Pollution generation rate 
  k_pollution_decay = 0.150 # Pollution decay rate
)
#set time steps and integration method 
time <- seq(0, 200, by = 0.01)

#set integration method
int_method <- "euler"

#create experimental design  

library(lhs)
set.seed(55555)
sample.size<-100
Xs<-c(
   'k_resources:X', 
   'ef_economy_resources_on_prod:X', 
   'ef_bureaucracy_on_prod:X', 
   'k_deprec:X',
   'ef_pollution_on_depreciation:X',
   'k_bureaucracy:X', 
   'ef_economy_on_bureaucracy:X', 
   'k_decay_bureaucracy:X', 
   'ef_pollution_on_bureaucracy:X', 
   'k_pollution:X',
   'k_pollution_decay:X' )


Exp <- randomLHS(sample.size, length(Xs)) |> 
  as.data.frame() |> 
  lapply(qunif, min = 0.5, max = 1.5) |> 
  as.data.frame()

colnames(Exp) <- Xs
Exp$Run.ID<-1:nrow(Exp)

#apply experimental design 
library(deSolve)
out_all<- apply(Exp,1,function(x) { p_x <- c(x['k_resources:X'], 
                                                    x['ef_economy_resources_on_prod:X'], 
                                                    x['ef_bureaucracy_on_prod:X'], 
                                                    x['k_deprec:X'],
                                                    x['ef_pollution_on_depreciation:X'],
                                                    x['k_bureaucracy:X'], 
                                                    x['ef_economy_on_bureaucracy:X'], 
                                                    x['k_decay_bureaucracy:X'], 
                                                    x['ef_pollution_on_bureaucracy:X'], 
                                                    x['k_pollution:X'],
                                                    x['k_pollution_decay:X']); 
                                          parameters <-p_0*p_x; 
                                          out <- data.frame(ode(y = state, times = time, func = bardis_model, parms = parameters, method=int_method))#;
                                          out <- subset(out,time%in%seq(0,max(out$time),0.2));
                                          out$Run.ID <- x['Run.ID'];
                                          out
                                         } )

out_all <- do.call("rbind",out_all)

#print database with experiment
write.csv(out_all,paste0(root,"tableau/bardis_ensamble.csv"),row.names=FALSE)

#print experimental design 
write.csv(Exp,paste0(root,"tableau/exp_design.csv"),row.names=FALSE)
