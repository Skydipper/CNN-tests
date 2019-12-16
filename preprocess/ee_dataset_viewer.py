import ee
from preprocess import ee_collection_specifics
from IPython.display import display, Image

class ee_dataset:
    
    def __init__(self, point, buffer, startDate, stopDate, scale, collection):
        """
        Class used to display the datasets from Earth Engine
        Parameters
        ----------
        points : list
            A list of two [x,y] coordinates with the centers of the area of interest.
        buffer : number
            Buffer in meters
        startDate : string
        stopDate : string
        scale: number
            Pixel size in meters.
        collection: string
            Name of each collection.

        """
        
        self.point = point
        self.buffer = buffer
        self.startDate = startDate
        self.stopDate = stopDate       
        self.scale = scale 
        self.collection = collection
        
        # Image Collection
        self.image_collection = ee_collection_specifics.ee_collections(self.collection)
 
        # Bands
        self.bands = ee_collection_specifics.ee_bands(self.collection)
    
        # Normalized Difference bands
        self.normDiff_bands = ee_collection_specifics.normDiff_bands(self.collection)
        
        # Normalized Difference band names
        self.normDiff_band_names = ee_collection_specifics.normDiff_band_names(self.collection)
        
        # Visualization parameters
        self.vizz_params = ee_collection_specifics.vizz_params(self.collection)
            
        # Area of Interest
        self.geom = ee.Geometry.Point(self.point).buffer(self.buffer)
        self.region = self.geom.bounds().getInfo()['coordinates']
        
        ## Composite
        self.image = ee_collection_specifics.Composite(self.collection)(self.image_collection, self.startDate, self.stopDate, self.geom)
        
        ## Calculate normalized Difference
        if self.normDiff_bands:
            for nB, normDiff_band in enumerate(self.normDiff_bands):
                image_nd = self.image.normalizedDifference(normDiff_band).rename(self.normDiff_band_names[nB])
                ## Concatenate images into one multi-band image
                if nB == 0:
                    self.image = ee.Image.cat([self.image.select(self.bands), image_nd])
                else:
                    self.image = ee.Image.cat([self.image, image_nd])
        else:
            self.image = self.image.select(self.bands)
           
        ## Choose the scale
        self.image = self.image.reproject(crs='EPSG:4326', scale=self.scale)
            
        ## Change data type
        if ee_collection_specifics.Dtype(self.collection):
            self.image = ee_collection_specifics.Dtype(self.collection)(self.image)

         
    def display_image(self):
        ## Display image
        if self.vizz_params:
            for i in range(len(self.vizz_params)):
                image = self.image.visualize(**self.vizz_params[i])
        
                visual = Image(url=image.getThumbUrl({'region':self.region}))
    
                display(visual)

            

                
           
            
            