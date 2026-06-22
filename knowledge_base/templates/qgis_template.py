from qgis.core import QgsVectorLayer, QgsProject
import processing

input_layer = QgsVectorLayer('{input_file}', 'input', 'ogr')
if not input_layer.isValid():
    raise ValueError('Invalid input layer')

# TODO: processing.run('native:...', params)

print('QGIS template ready')
