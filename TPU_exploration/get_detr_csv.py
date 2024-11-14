import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from transformers import DetrForObjectDetection

# Load the pretrained model
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Input image dimensions
input_height = 260
input_width = 260

class ConvolutionalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1)
        self.dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2, stride=1)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        
        x1 = self.conv1(x)
        
        pad3 = max(0, (3 - h) // 2)
        x3 = F.pad(x, (pad3, pad3, pad3, pad3), mode='replicate')
        x3 = self.conv3(x3)
        
        pad5 = max(0, (5 - h) // 2)
        x5 = F.pad(x, (pad5, pad5, pad5, pad5), mode='replicate')
        x5 = self.conv5(x5)
        
        pad_dilated = max(0, (5 - h) // 2)
        x_dilated = F.pad(x, (pad_dilated, pad_dilated, pad_dilated, pad_dilated), mode='replicate')
        x_dilated = self.dilated_conv(x_dilated)
        
        x = x1 + x3 + x5 + x_dilated
        
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return self.dropout(x)

class AttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.global_pool(x)
        weights = self.fc(weights)
        return x * weights

class QueryFilters(nn.Module):
    def __init__(self, in_channels, num_queries):
        super().__init__()
        self.queries = nn.Conv2d(in_channels, num_queries, kernel_size=1, stride=1)

    def forward(self, x):
        return self.queries(x)

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.bbox_head = nn.Conv2d(in_channels, 4, kernel_size=1, stride=1)
        self.class_head = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        bboxes = self.bbox_head(x)
        class_scores = self.class_head(x)
        return bboxes, class_scores

def extract_layer_info(layer, current_height, current_width):
    if not isinstance(layer, nn.Conv2d):
        return None, current_height, current_width

    # Calculate required padding to ensure output dimensions are positive
    effective_kernel_h = layer.kernel_size[0] + (layer.kernel_size[0] - 1) * (layer.dilation[0] - 1)
    effective_kernel_w = layer.kernel_size[1] + (layer.kernel_size[1] - 1) * (layer.dilation[1] - 1)
    
    # Calculate and add padding to IFMAP in case weight kernel has larger dimensions
    required_pad_h = max(0, (effective_kernel_h - current_height + layer.stride[0] - 1) // 2)
    required_pad_w = max(0, (effective_kernel_w - current_width + layer.stride[1] - 1) // 2)
    padded_height  = current_height + 2 * required_pad_h
    padded_width   = current_width + 2 * required_pad_w

    layer_info = {
        "Layer name": layer.__class__.__name__,
        "IFMAP Height": padded_height,
        "IFMAP Width": padded_width,
        "Filter Height": layer.kernel_size[0],
        "Filter Width": layer.kernel_size[1],
        "Channels": layer.in_channels,
        "Num Filter": layer.out_channels,
        "Strides": layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
    }
    
    # Calculate output dimensions using padded input
    new_height = ((padded_height + 2 * layer.padding[0] - 
                  layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0]) + 1
    new_width = ((padded_width + 2 * layer.padding[1] - 
                 layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1]) + 1

    # Ensure positive dimensions
    new_height = max(1, new_height)
    new_width = max(1, new_width)

    return layer_info, new_height, new_width
    

# Create custom Transformer layers that can be expressed as a set of convolutions
model.conv_encoder = ConvolutionalEncoder(256, 512)
model.attention_pooling = AttentionPooling(512)
model.query_filters = QueryFilters(512, num_queries=100)
model.detection_head = DetectionHead(512, num_classes=model.config.num_labels)

# Extract layer information
layer_info_list = []
current_height, current_width = input_height, input_width
for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d):
        layer_info, current_height, current_width = extract_layer_info(layer, current_height, current_width)
        if layer_info:
            layer_info["Layer name"] = name
            layer_info_list.append(layer_info)

# Write DETR architecture to .csv file for Scale-Sim
header = ["Layer name", "IFMAP Height", "IFMAP Width", "Filter Height", "Filter Width", "Channels", "Num Filter", "Strides"]
with open('detr.csv', 'w', newline='') as csvfile:
    # Write each row with trailing comma for proper Scale-Sim parsing 
    header_str = ','.join(header) + ','
    csvfile.write(header_str + '\n')
    for layer_info in layer_info_list:
        row_values = []
        for field in header:
            value = layer_info[field]
            row_values.append(str(value))
        row_str = ','.join(row_values) + ','
        csvfile.write(row_str + '\n')

print("\nDETR layer information written to detr.csv")
