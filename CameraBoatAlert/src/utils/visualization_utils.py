# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

# Imports
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import collections
import numpy as np
import os
import matplotlib.path as mpltPath

current_path = os.getcwd()
        
# phuongtu
def draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, capture_area=[]
                                     , color='red', thickness=4, display_str_list=(), use_normalized_coordinates=True):
    """Adds a bounding box to an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box in normalized coordinates (same below).
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    is_vehicle_detected, predicted_direction = draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, capture_area, 
                                                                          color, thickness, display_str_list, use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))
    return is_vehicle_detected, predicted_direction

# phuongtu
def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, capture_area=[], color='red', thickness=4, display_str_list=(), use_normalized_coordinates=True):
    """Adds a bounding box to an image.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
    is_vehicle_detected = False
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
              (right, top), (left, top)], width=thickness, fill=color)
              
    predicted_direction = ""

    if(inside_polygon(left, top, capture_area) and inside_polygon(right, top, capture_area)):
        predicted_direction = "up"
        is_vehicle_detected = True
    else:
        if(inside_polygon(left, bottom, capture_area) and inside_polygon(right, bottom, capture_area)):
            predicted_direction = "down"
            is_vehicle_detected = True

    try:
        font = ImageFont.truetype('arial.ttf', 16)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_list[0] = display_str_list[0]
  
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width,text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin), display_str, fill='black', font=font)
        text_bottom -= text_height - 2 * margin
        return is_vehicle_detected, predicted_direction

# phuongtu
def visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index, 
                                              targeted_objects=None, capture_area=[],
                                              use_normalized_coordinates=False, max_boxes_to_draw=20, min_score_thresh=.5):
  
    if targeted_objects is not None:
        arr_category = category_index.values()
        obj_len = len(targeted_objects)
        for i in range(0, obj_len):
            for cat in arr_category:
                if cat.get("name") == targeted_objects[i]:
                    targeted_objects[i] = cat.get('id')
                    break
          
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.

    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if targeted_objects is not None:
            if classes[i] not in targeted_objects:
                continue
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if scores is None:
                box_to_color_map[box] = 'black'
            else:
                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]['name']         
                else:
                    class_name = 'N/A'              
                display_str = '{}: {}%'.format(class_name, int(100 * scores[i]))

                box_to_display_str_map[box].append(display_str)
                box_to_color_map[box] = 'BurlyWood'
    
    is_vehicle_detected = False
    predicted_direction = ""
    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        
        display_str_list = box_to_display_str_map[box]

        # we are interested just boat
        if (("cell phone" in display_str_list[0]) or ("boat" in display_str_list[0])):
            is_detected, predicted_direction = draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, capture_area,
                                                                                color=color, thickness=4, display_str_list=box_to_display_str_map[box],
                                                                                use_normalized_coordinates=use_normalized_coordinates)
        if is_detected:
            is_vehicle_detected = is_detected
            
    return is_vehicle_detected, predicted_direction

def inside_polygon(x, y, polygon):
    path = mpltPath.Path(polygon)
    return path.contains_point((x, y))