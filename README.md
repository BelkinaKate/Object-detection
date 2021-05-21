# Распознавание объектов на изображениях
Используя модели YOLOv3, YOLO tiny, RetinaNet, SSD и Faster RCNN


**YOLOv3** — это усовершенствованная версия архитектуры YOLO. Она состоит из 106-ти 
свёрточных слоев и лучше детектирует небольшие объекты по сравнению с её предшествиницей YOLOv2. 
Основная особенность YOLOv3 состоит в том, что на выходе есть три слоя каждый из которых расчитан на обнаружения объектов разного размера.

**YOLOv3-tiny** — обрезанная версия архитектуры YOLOv3, состоит из меньшего количества слоев (выходных слоя всего 2). Она более хуже предсказывает
мелкие объекты и предназначена для небольших датасетов. Но, из-за урезанного строения, веса сети занимают небольшой объем памяти (~35 Мб) и она 
выдает более высокий FPS. Поэтому такая архитектура предпочтительней для использования на мобильном устройстве.

**RetinaNet** - это одноступенчатый детектор, который использует Feature Pyramid Network (FPN) и Focal loss для обучения.
Особенность этой модели заключается в том, что информация с карт, получаемых на выходе сверточных слоев,передается на 
вход деконволюционных слоев, где для каждого строится свой набор ограничивающих прямоугольников, которые описывают 
местоположении объекта на изображении.Данная архитектура хорошо себя зарекомендовала на задачах детекции различного рода объектов. 

**SSD** (Single Shot MultiBox Detector) – используются наиболее удачные «хаки» архитектуры YOLO (например, non-maximum suppression) 
и добавляются новые, чтобы нейросеть быстрее и точнее работала. Отличительная особенность: различение объектов за один прогон 
с помощью заданной сетки окон (default box) на пирамиде изображений. Пирамида изображений закодирована в сверточных тензорах 
при последовательных операциях свертки и пулинга (при операции max-pooling пространственная размерность убывает). 
Таким образом определяются как большие, так и маленькие объекты за один прогон сети.

**Faster R-CNN** - быстрейшая модель типа R-CNN.
Серия архитектур на основе R-CNN (Regions with Convolution Neural Networks features): R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN. 
Для обнаружения объекта на изображении с помощью механизма Region Proposal Network (RPN) выделяются ограниченные регионы (bounding boxes).
В архитектуре R-CNN есть явные циклы «for» перебора по ограниченным регионам, всего до 2000 прогонов через внутреннюю сеть AlexNet. 
Из-за явных циклов «for» замедляется скорость обработки изображений. Количество явных циклов, прогонов через внутреннюю нейросеть, 
уменьшается с каждой новой версией архитектуры.
