# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in /usr/local/Cellar/android-sdk/24.3.3/tools/proguard/proguard-android.txt
# You can edit the include path and order by changing the proguardFiles
# directive in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# Add any project specific keep options here:
# Zachowaj klasy React Native
-keep class com.facebook.** { *; }
-dontwarn com.facebook.**

# Zachowaj klasy do komunikacji sieciowej (fetch, FormData)
-keep class okhttp3.** { *; }
-dontwarn okhttp3.**
-keep class okio.** { *; }
-dontwarn okio.**

# Zachowaj klasy react-native-blob-util (używasz w HomeScreen.tsx)
-keep class com.RNFetchBlob.** { *; }
-dontwarn com.RNFetchBlob.**

# Zachowaj klasy react-native-image-picker
-keep class com.imagepicker.** { *; }
-dontwarn com.imagepicker.**

# Zachowaj klasy powiadomień (react-native-push-notification)
-keep class com.google.firebase.** { *; }
-dontwarn com.google.firebase.**
