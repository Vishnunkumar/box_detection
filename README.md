# box_detection
Detect tables and check boxes in a document using OpenCV

## Sample code 
```
import box_detection.detect as det
df = det.get_boxes('input_data/images/sample2.jpg', 10, 220, 15, "table", 'input_data/images/out3.png')
```
- Input arguements are as follows :
	- image_path: path of the image file
	- lw : length of the kernel
	- threshold_min : min value for binarization
	- cc : connectivity for the pixel separation
	- mode : either "table" or "check boxes"
	- output_path : saving image path
