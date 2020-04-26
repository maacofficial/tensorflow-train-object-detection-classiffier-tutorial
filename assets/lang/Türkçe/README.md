<h1>Tensorflow İle Birlikte Windows 10'da Kendi Modelimizi Eğitme</h1>


<a href="https://www.youtube.com/watch?v=Wp9VDu7cpyk" target="_blank"><img src="https://3.bp.blogspot.com/-zN9ZbZrBtVc/XJMwMUlyVDI/AAAAAAAABIM/brSGzSlHmNMIFYZdXnMZhCzEBxwZf0bfACEwYBhgL/s1600/Object%2BDetection%2BTensorflow.jpg" alt="How to train own object detection classifier using Tensorflow (CPU) on Windows 10"></a>
<p>Eğer destek olmak için yeni bir çeviri eklemek isterseniz istek oluşturabilirsiniz.</p>

<p>Kendi modelinizi Windows 10/8 ve 7 de eğitebilirsiniz. Bu proje Tensorflow 1.5 sürümü kullanılarak hazırlanmıştır. Eğer siz farklı bir sürümle yapmak isterseniz uyumluluk sorunundan dolayı hatalarla karşılaşabilirsiniz.</p>

<p>Ayrıca bu projenin nasıl yapıldığına dair bir video da çektim.</p>


<h2>Gerekli Açıklamalar</h2>

<p>Siz TensorFlow (GPU) yerine TensorFlow (CPU) kullanabilirsiniz. Bunu yapmak için "pip install --ignore-installed --upgrade tensorflow==1.5.0" komutu yerine "pip install --ignore-installed --upgrade tensorflow-gpu==1.5.0" komutunu çalıştırmanız gerekir.Aynı zamanda ekran kartınız için CUDA ve cuDNN driverlarını kurmanız gerekir.</p>

<p>Eğer eğitmek için GPU kullanıyorsanız,Nvidia ekran kartına sahip olmanız gerekir. Ayrıca grafik kartınız minimum gereksinimleri karşılamalıdır. TensorFlow minimum 3.5 hesaplama kapasitesine(CC) ihtiya. duymaktadır. <a href="https://developer.nvidia.com/cuda-gpus" target="_blank">Bütün Ekran Kartları İçin Hesaplama Kapasitesi Listesi</a></p>
<p>Eğer ekran kartınız orta seviyeyse, bu işlem 3 saat kadar alıcaktır. Ve <a href="https://www.tensorflow.org/install/source_windows#gpu" target="_blank">burada</a> TensorFlow sürümleriyle uyumlu CUDA versionları bulunmaktadır.</p>

<h2>Adımlar</h2>

<h3>1.Anaconda Yükleme</h3>

<p>İlk olarak,ücretsiz ve açık kaynaklı bir dağıtım olan Anaconda'yı <a href="https://www.anaconda.com/distribution/#download-section" target="_blank">bu linkten</a> indirelim.</p>

<p>Adımları videodan takip edebilirsiniz.</p>


<h3>2.TensorFlow Klasörünü Oluşturma</h3>

Eğer farklı bir TensorFlow kütüphanesiyle çalışmak istemiyorsanız, direk olarak bu projeyi indirip çıkartabilirsiniz. <a href="https://codeload.github.com/maacofficial/tensorflow-train-object-detection-classiffier-tutorial/zip/master" target="_blank">Bu linkten</a>. Ve bir sonraki adıma geçelim.

<p>2.1. C:/ dizininda "tensorflow1" adında klasör oluşturalım. Bu klasör bizim model,training ve object_detection klasörlerimizi barındırıcak. Kısacası bu klasör içinde çalışacağız.</p>

<p>2.2. Aşağıdaki linklerden sizin tensorflow sürümünüze uygun olan model dosyalarını indiriniz.Biz bu projede TensorFlow 1.5 sürümünü kullanacağız.</p>

<table>
<thead>
<tr>
<th>TensorFlow version</th>
<th>GitHub Models Repository</th>
</tr>
</thead>
<tbody>
 <tr>
<td>TF v1.4</td>
<td><a href="https://github.com/tensorflow/models/tree/1f34fcafc1454e0d31ab4a6cc022102a54ac0f5b">https://github.com/tensorflow/models/tree/1f34fcafc1454e0d31ab4a6cc022102a54ac0f5b</a></td>
</tr>
<tr>
<td>TF v1.5</td>
<td><a href="https://github.com/tensorflow/models/tree/d90d5280fea4a5303affc1e28af505d8292d84b8">https://github.com/tensorflow/models/tree/d90d5280fea4a5303affc1e28af505d8292d84b8</a></td>
</tr>
 <tr>
<td>TF v1.6</td>
<td><a href="https://github.com/tensorflow/models/tree/4c05414826e87f3b8ef0534862748e4b7fcd1ec7">https://github.com/tensorflow/models/tree/4c05414826e87f3b8ef0534862748e4b7fcd1ec7</a></td>
</tr>
<tr>
<td>TF v1.7</td>
<td><a href="https://github.com/tensorflow/models/tree/adfd5a3aca41638aa9fb297c5095f33d64446d8f">https://github.com/tensorflow/models/tree/adfd5a3aca41638aa9fb297c5095f33d64446d8f</a></td>
</tr>
<tr>
<td>TF v1.8</td>
<td><a href="https://github.com/tensorflow/models/tree/abd504235f3c2eed891571d62f0a424e54a2dabc">https://github.com/tensorflow/models/tree/abd504235f3c2eed891571d62f0a424e54a2dabc</a></td>
</tr>
<tr>
<td>TF v1.9</td>
<td><a href="https://github.com/tensorflow/models/tree/d530ac540b0103caa194b4824af353f1b073553b">https://github.com/tensorflow/models/tree/d530ac540b0103caa194b4824af353f1b073553b</a></td>
</tr>
<tr>
<td>TF v1.10</td>
<td><a href="https://github.com/tensorflow/models/tree/b07b494e3514553633b132178b4c448f994d59df">https://github.com/tensorflow/models/tree/b07b494e3514553633b132178b4c448f994d59df</a></td>
</tr>
<tr>
<td>TF v1.11</td>
<td><a href="https://github.com/tensorflow/models/tree/23b5b4227dfa1b23d7c21f0dfaf0951b16671f43">https://github.com/tensorflow/models/tree/23b5b4227dfa1b23d7c21f0dfaf0951b16671f43</a></td>
</tr>
<tr>
<td>TF v1.12</td>
<td><a href="https://github.com/tensorflow/models/tree/r1.12.0">https://github.com/tensorflow/models/tree/r1.12.0</a></td>
</tr>
<tr>
<td>TF v1.13</td>
<td><a href="https://github.com/tensorflow/models/tree/r1.13.0">https://github.com/tensorflow/models/tree/r1.13.0</a></td>
</tr>
<tr>
<td>Latest version</td>
<td><a href="https://github.com/tensorflow/models">https://github.com/tensorflow/models</a></td>
</tr>
</tbody>
</table>
<p>2.3. "C:/" dizininin içinde oluşturduğumuz "tensorflow1" klasörünün içine indirdiğimiz model dosyalarını çıkartalım ve çıkardığımız bu klasörü "models" olarak yeniden adlandıralım.</p>

<p>2.4. "Faster-RCNN-Inception-V2-COCO" modelini TensorFlow'un model zoo kısmından indirelim: <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md">link</a>.</p>

<p>2.5. "faster_rcnn_inception_v2_coco_2018_01_28.tar.gz" isimli dosyayı "C:\tensorflow1\models\research\object_detection\" klasörünün içerisine çıkaralım.</p>

<p>2.6. "Edje Electronics repository" dosyalarını indirelim: <a href="https://codeload.github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/zip/master">link</a>.</p>

<p>2.7. İndirdiğimiz "TensorFlow-Object-Detection-API.zip" isimli dosyayı "C:\tensorflow1\models\research\object_detection\" klasörüne çıkartalım.</p>

<p>2.8. Son olarak "object_detection" isimli klasör aşağıdaki resimdeki gibi gözükmelidir.</p>
<div><img src="https://github.com/maacofficial/tensorflow-train-object-detection-classiffier-tutorial/blob/master/assets/repos.png" width="400" height="557">
</div>


<h3>3. Anaconda Ortamını Oluşturma ve Kurma</h3>
<p>3.1 Komut istemcisini yönetici olarak çalıştırın.</p>
<p>3.2 Daha sonra bu komutu çalıştırın: <code>C:\>conda create -n tensorflow1 pip python=3.5</code></p>
<p><code>C:\>conda activate tensorflow1</code></p>
<p>3.3 TensorFlow 1.5 sürümünü yükleyelim: <code>C:\>pip install --ignore-installed --upgrade tensorflow==1.5.0</code></p>
<p> Eğer TensorFlow GPU 1.5 sürümünü yükleyecekseniz: <code>C:\>pip install --ignore-installed --upgrade tensorflow-gpu==1.5.0</code></p>
<p>3.5 Aşağıdaki komutları girerek gerekli paketleri yükleyelim;<p>
<p><code>C:\>conda install -c anaconda protobuf</code><p>
<p><code>C:\>pip install pillow lxml cython jupyter matplotlib pandas opencv-python</code><p>
<p>3.6 Çevre Değişkenlerini kurmak için: 
<pre><code>C:\>set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim</code></pre><p>

 
<h3>4. Protobufları derleyelim ve setup.py dosyasını çalıştıralım</h3>
Komut isteminde aşağıdaki komutu çalıştırarak "research" klasörüne gidin:
<pre><code>C:\>cd C:\tensorflow1\models\research</code></pre>

Daha sonra devamındaki kodları çalıştırın:
<pre><code>protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto</code></pre>
<pre><code>(tensorflow1) C:\tensorflow1\models\research>python setup.py build</code></pre>
<pre><code>(tensorflow1) C:\tensorflow1\models\research>python setup.py install</code></pre>

<h3>5. Fotoğraf Toplama ve Resimleri Etiketleme</h3>

<p>Resim çekmek için telefonunuzun kamerasını kullanabilirsiniz. Ama çektiğiniz fotoğrafların boyutunun 200KB ve çözünürlüklerinin ise 720x1280 değerlerinden düşük olmasına dikkat edin. Resimleri topladıktan sonra,resimlerin %20'sini "C:\tensorflow1\models\research\object_detection\images\test\" klasörüne ve %80'ini ise "C:\tensorflow1\models\research\object_detection\images\train\" klasörüne atın. Son olarak resimlerinizin çeşitli şekillerde çekildiğine dikkat edin.</p>

Bu adımda ise resimleri etiketlemek için "LabelImg" aracını indireceğiz: <a href="https://tzutalin.github.io/labelImg/">Bu linkten</a>.


<h3>6. Eğitim Verilerini Oluşturma</h3>

<p>İlk olarak resimlerin xml verilerini csv dosyasına çevireceğiz. Bunu yapmak için:</p>
<pre><code>(tensorflow1) C:\tensorflow1\models\research\object_detection>python xml_to_csv.py</code></pre>

<p>Bundan sonra, generate_tfrecord.py dosyasını notepad ile düzenleyelim. Etiket haritasını kendi haritanızla değiştirin.</p>


<pre><code># TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == '200TL':
        return 1
    elif row_label == '5TL':
        return 2
    else:
        None</code></pre>
 <p>TO:</p>
 
 <pre><code># TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'cat':
        return 1
    elif row_label == 'dog':
        return 2
    elif row_label == 'bird':
        return 3
    elif row_label == 'person':
        return 4
    else:
        None</code></pre>
 <p>Daha sonra,veri dosyalarını oluşturmak için "C:\tensorflow1\models\research\object_detection" klasörünün içerisinde aşağıdaki komutları çalıştırın:</p>
 
  <pre><code>python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record</code></pre>

<h3>7. Etiket haritasını ve eğitme ayarlarını düzenleme</h3>
<p>"C:\tensorflow1\models\research\object_detection\training\labelmap.pbtxt" dosyasını notepad ile düzenleyin.</p>


 <pre><code>item {
  id: 1
  name: '200TL'
}

item {
  id: 2
  name: '5TL'
}</code></pre>

To:

 <pre><code>item {
  id: 1
  name: 'cat'
}

item {
  id: 2
  name: 'dog'
}
item {
  id: 3
  name: 'bird'
}
item {
  id: 4
  name: 'person'
}</code></pre>

<p>Bundan sonra "C:\tensorflow1\models\research\object_detection\training\faster_rcnn_inception_v2_pets.config" dosyasını notepad ile düzenleyin.</p>

<p>a. Satır 9 - num_classes değerini kameraya algılatmak istediğiniz nesne sayısı ile değiştirin.Örneğin; cat,dog,bird and person "num_classes:4" </p>
<p>b. Satır 110 - fine_tune_checkpoint değiştirin:
 <pre><code>
fine_tune_checkpoint: "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"</code></pre>
</p>
<p>c. Satır 126-128 - train_input_reader bölümündeki bazı değişkenleri değiştirin:
 <pre><code>
train_input_reader: {
  tf_record_input_reader {
    input_path: "C:/tensorflow1/models/research/object_detection/train.record"
  }
  label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
}
</code></pre>
</p>
<p>d. Satır 140-142 - eval_input_reader bölümündeki bazı değişkenleri değiştirin:
 <pre><code>
  tf_record_input_reader {
    input_path: "C:/tensorflow1/models/research/object_detection/test.record"
  }
  label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
  shuffle: false
  num_readers: 1
}
</code></pre>
</p>

<p>e. Satır 132 -  num_examples değerini "C:\tensorflow1\models\research\object_detection\images\test" klasörünün içindeki resim sayısıyla değiştirin:
<code>num_examples: 89</code>
</p>

<p>f. Son olarak dosyalarınızdaki değişiklikleri kaydedin ve çıkın.</p>

<h3>8. train.py dosyasını çalıştırma</h3>

<p>Eğer TensorFlow'un daha yeni bir sürümünü kullanıyorsanız ,train.py dosyası C:\tensorflow1\models\research\object_detection\legacy" klasörünün içinde bulunur. Bu dosyayı object_detection klasörüne kopyalayabilirsiniz.</p>

<p>İşte en aksiyonlu an! train.pydosyasını çalıştıralım. Belirtilen komutu "C:\tensorflow1\models\research\object_detection" klasörünün içinde çalıştırın:</p>
<pre><code>python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
</code></pre>

<p>Eğitim başladığında aşağıdaki resimdeki gibi gözükür:</p>
<div><img src="https://github.com/maacofficial/tensorflow-train-object-detection-classiffier-tutorial/blob/master/assets/screenshot.png">
</div>

<h3>9. Modelimizi çıkartma</h3>

<p>Eğitim işlemi tamamlandığında,en son model dosyamızı oluşturmalıyız(.pb).Aşağıdaki komutta "XXXX" değerini "C:\tensorflow1\models\research\object_detection\training" klasöründeki en büyük sayı ile değiştirin. Daha sonra belirtilen komutu "C:\tensorflow1\models\research\object_detection" klasörünün içinde çalıştırın:</p>
<pre><code>python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
</code></pre>

<p>Oluşturulan model dosyası "C:\tensorflow1\models\research\object_detection\inference_graph" klasörünün içinde bulunmaktadır.</p>

<h3>10. Kendi modelimizi kullanma</h3>
<p>Eğer WebCam ile nesneleri tanımak istiyorsanız "C:\tensorflow1\models\research\object_detection" klasörünün içinde yer alan "Object_detection_webcam.py" dosyasını çalıştırın."Object_detection_webcam.py" dosyasını çalıştırmadan önce ,"NUM_CLASSES = 2" bölümünü düzenleyin.Son olarak aşağıdaki komutu çalıştırın:</p>
<pre><code>python Object_detection_webcam.py</code></pre>

<h4>Hayalleriniz gerçekleşinceye kadar hayal etmeye devam edin!!!</h4>
