# Define the system of differential equations
bardis_model <- function(time, state, parameters) {
  with(as.list(c(state, parameters)), {
    
    # Flows

    #Resources 
    resource_inflow <- k_resources * Resources
    extractive_pollution <- k_pollution * Economy * Resources 

    #Economy 
    production <- ef_economy_resources_on_prod * Resources * Economy + ef_bureaucracy_on_prod * Bureaucracy  
    depreciation <- k_deprec * Economy + ef_pollution_on_depreciation * Pollution 
    
    #Bureaucracy 
    bureaucracy_creation <- ef_economy_on_bureaucracy * Economy + k_bureaucracy * Bureaucracy
    bureaucracy_decay <- k_decay_bureaucracy  * Bureaucracy + ef_pollution_on_bureaucracy * Pollution 

    #Pollution 
    pollution_abtement <- k_pollution_decay * Pollution

    # Rate of change
    dResources <- ifelse(Resources > 0, resource_inflow - production - extractive_pollution, 0)
    dEconomy <- ifelse(Economy > 0, production - depreciation - bureaucracy_creation, 0)
    dBureaucracy <- ifelse(Bureaucracy > 0, bureaucracy_creation - bureaucracy_decay, 0)
    dPollution <- ifelse(Pollution > 0, depreciation + bureaucracy_decay + extractive_pollution - pollution_abtement, 0)   
    list(c(dResources, dEconomy, dBureaucracy, dPollution))
  })
}