package com.example.imageclassifier;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.imageclassifier.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
    ImageView imageView;
    Button camera, gallery;
    TextView result;
    int imgSize = 32;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imgView);
        camera = findViewById(R.id.clkpic);
        result = findViewById(R.id.result);
        gallery = findViewById(R.id.gallerypic);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(checkSelfPermission(Manifest.permission.CAMERA)== PackageManager.PERMISSION_GRANTED){
                    Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(intent, 3);
                }
                else{
                    requestPermissions(new String[]{Manifest.permission.CAMERA},100);
                }
            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                    Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                    startActivityForResult(intent, 1);

            }
        });


    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode==RESULT_OK){
            if(requestCode==3){
                assert data != null;
                Bitmap bitmap = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(bitmap.getWidth(), bitmap.getHeight());
                bitmap = ThumbnailUtils.extractThumbnail(bitmap, dimension, dimension);
                imageView.setImageBitmap(bitmap);

                bitmap = Bitmap.createScaledBitmap(bitmap,imgSize,imgSize, false);
                classify(bitmap);
            }
            else{
                Uri uri = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                }
                catch (IOException e) {
                    throw new RuntimeException(e);
                }
                imageView.setImageBitmap(image);
                image = Bitmap.createScaledBitmap(image,imgSize,imgSize, false);
                classify(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void classify(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*imgSize*imgSize*3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intVal = new int[imgSize*imgSize];
            image.getPixels(intVal,0, image.getWidth(),0,0,image.getWidth(), image.getHeight());

            int pixel = 0;
            for(int i=0;i<imgSize;i++){
                for(int j=0;j<imgSize;j++){
                    int val = intVal[pixel++];
                    byteBuffer.putFloat(((val>>16)& 0xFF)*(1.f/1));
                    byteBuffer.putFloat(((val>>8)& 0xFF)*(1.f/1));
                    byteBuffer.putFloat((val& 0xFF)*(1.f/1));
                }
            }
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for(int i=0;i<confidences.length;i++){
                if(confidences[i]>maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;

                }
            }

            String[] classes = {"Apple", "Banana", "Orange"};
            result.setText(classes[maxPos]);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

    }


}