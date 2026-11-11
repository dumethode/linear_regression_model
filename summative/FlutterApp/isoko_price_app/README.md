# Isoko Price Check (Flutter app)

One page app that sends the year, month, market, product and price type to the API and shows the estimated price.

```bash
cd summative/FlutterApp/isoko_price_app
flutter create --platforms=android,ios --project-name isoko_price_app .   # only the first time, adds the android/ and ios/ folders
flutter pub get
flutter run            # with an Android emulator or a phone connected
flutter test           # widget tests
```

The API address is set in `lib/api_service.dart` (`kApiBaseUrl`).
