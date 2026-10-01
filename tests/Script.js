db.getCollection("predictor_models_cache").count({})

db.getCollection("predictor_models_cache").find({"prediction.modelResults.predictionError": { $ne: null }})

db.getCollection("predictor_models_cache").count({"prediction.modelResults.predictionError": { $ne: null }})

db.getCollection("predictor_models_cache").find(
{"prediction.modelResults.predictionError": { $nin: [null, "Descriptor calculation failed"] }}, 
{"prediction.modelResults.predictionError": 1, "prediction.chemicalIdentifiers.smiles": 1})


db.getCollection("predictor_models_cache").find({"prediction.modelResults.predictionError": { $ne: null }},
    {"prediction.modelResults.predictionError": 1, "prediction.chemicalIdentifiers.smiles": 1})

db.getCollection("predictor_models_cache").countDocuments({})

db.getCollection("predictor_models_cache").countDocuments({"prediction.modelResults.predictionError": { $ne: null }})