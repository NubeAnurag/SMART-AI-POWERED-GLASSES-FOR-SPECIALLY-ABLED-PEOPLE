# Comprehensive Multi-Object Dataset Guide

## 🎯 Target Objects (15 classes):
- carpet
- mat
- rug
- window
- aquarium
- pen
- photo_frame
- picture_frame
- microwave
- ceiling_fan
- table_fan
- fan
- idol
- split_ac
- window_ac

## 📊 Recommended Dataset Sizes:
- **Training**: 150-300 images per object (1,800-3,600 total)
- **Validation**: 30-60 images per object (360-720 total)
- **Test**: 15-30 images per object (180-360 total)

## 📸 Image Collection Strategy:

### 🏠 Floor Items (carpet, mat, rug):
- Different types: Persian, modern, shag, outdoor
- Various colors and patterns
- Different room contexts: living room, bedroom, kitchen
- Different angles: overhead, side view, corner view

### 🪟 Windows:
- Different styles: casement, sliding, bay, picture
- Various frames: wooden, metal, vinyl
- Different contexts: home, office, car
- Various lighting: day, night, curtains open/closed

### 🐠 Aquariums:
- Different sizes: small, medium, large
- Various types: freshwater, saltwater, planted
- Different shapes: rectangular, round, bow-front
- Various contents: fish, plants, decorations

### ✏️ Writing Tools (pen):
- Different types: ballpoint, fountain, gel, marker
- Various brands and colors
- Different contexts: desk, hand, pocket
- Various angles: side view, top view, writing position

### 🖼️ Photo Frames:
- Different styles: modern, vintage, ornate
- Various materials: wood, metal, plastic
- Different sizes: small, medium, large
- Various contexts: wall, table, shelf

### 🍽️ Kitchen Appliances (microwave):
- Different brands and models
- Various sizes and colors
- Different contexts: counter, built-in, office
- Various states: clean, in use, with food

### 💨 Fans (ceiling_fan, table_fan, fan):
- Different types: ceiling, table, floor, wall
- Various styles: modern, traditional, industrial
- Different sizes and blade counts
- Various contexts: home, office, outdoor

### 🏛️ Religious Items (idol):
- Different religions: Hindu, Buddhist, Christian, etc.
- Various materials: stone, metal, wood, ceramic
- Different sizes: small, medium, large
- Various contexts: temple, home, outdoor shrines

### ❄️ Air Conditioning (split_ac, window_ac):
- **split_ac**: Indoor and outdoor units, wall-mounted
- **window_ac**: Single unit mounted in window
- Different brands and models
- Various sizes and capacities
- Different contexts: home, office, commercial

## 🏷️ Labeling Guidelines:

### Class Mapping:
- **carpet**: Large floor coverings, rugs, carpets
- **mat**: Small floor mats, doormats, bath mats
- **rug**: Medium floor coverings, area rugs
- **window**: Any window type, glass panels
- **aquarium**: Fish tanks, terrariums, water containers
- **pen**: Writing instruments, markers, pencils
- **photo_frame**: Picture frames, photo displays
- **microwave**: Microwave ovens, food heating devices
- **ceiling_fan**: Fans mounted on ceiling
- **table_fan**: Small portable fans
- **fan**: Generic fan category
- **idol**: Religious statues, figurines, deities
- **split_ac**: Split air conditioning units (indoor + outdoor)
- **window_ac**: Window-mounted air conditioning units

## 📈 Expected Performance:
- **mAP50**: 75-85% (with good data quality)
- **False Positives**: <15%
- **Training Time**: 3-5 hours on M1 GPU
- **Model Size**: ~64MB

## 🚫 Avoid Overfitting:
- **Diverse backgrounds**: Different rooms, lighting, angles
- **Multiple brands/models**: Don't use same object repeatedly
- **Varied contexts**: Objects in different environments
- **Quality images**: Clear, well-lit, focused photos

## 🎨 Data Diversity Tips:
1. **Use your phone** to take photos of objects around you
2. **Ask friends/family** to contribute images
3. **Use online sources** (ensure you have rights)
4. **Vary lighting conditions**: bright, dim, natural, artificial
5. **Include negative samples**: scenes without target objects
