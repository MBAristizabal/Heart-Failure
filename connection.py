import pymongo
import settings

def connect(nombredb):

    # # conexión a base de datos
    # # Si no existe la base de datos la crea con el nombre que haya en nombredb
    bbdd = pymongo.MongoClient(settings.HOST, settings.PORT)

    # # Carga la base de datos en un Dataframe y lo devuelve
    db = bbdd[nombredb]

    return db