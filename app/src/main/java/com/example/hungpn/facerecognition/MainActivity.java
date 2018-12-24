package com.example.hungpn.facerecognition;

import android.content.ContextWrapper;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.ImageView;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.File;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Bitmap bmp = BitmapFactory.decodeFile("data/lena.jpg");
        //Bitmap bmp = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888);
        //ImageView imageView = (ImageView) findViewById(R.id.imageView);
        //InputStream is = getClass().getResourceAsStream("/drawable/" + "lena.jpg");
        //imageView.setBackgroundColor(Color.BLUE);
        //imageView.setImageDrawable(Drawable.createFromStream(is, ""));
       // imageView.setImageBitmap(bmp);

        //int rt = captureCamera()

        // Example of a call to a native method
        TextView tv = (TextView) findViewById(R.id.sample_text);
        tv.setText(stringFromJNI() + String.format("\nReturn: %d", 100));
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
    public native int faceRecognition();
    public native int captureCamera();
}
