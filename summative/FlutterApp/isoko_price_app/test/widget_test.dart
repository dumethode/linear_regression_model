import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:isoko_price_app/api_service.dart';
import 'package:isoko_price_app/main.dart';

void main() {
  testWidgets('page has five labelled fields and a Predict button', (tester) async {
    await tester.pumpWidget(const IsokoApp());
    expect(find.byType(TextField), findsNWidgets(5));
    for (final label in ['Year', 'Month', 'Market', 'Product', 'Price type']) {
      expect(find.text(label), findsOneWidget);
    }
    expect(find.text('Predict'), findsOneWidget);
  });

  testWidgets('missing and wrong values show an error in the result area', (tester) async {
    await tester.pumpWidget(const IsokoApp());
    await tester.enterText(find.byKey(const Key('yearField')), 'abc');
    await tester.enterText(find.byKey(const Key('monthField')), '13');
    await tester.tap(find.byKey(const Key('predictButton')));
    await tester.pump();
    final error = tester.widget<Text>(find.byKey(const Key('errorText'))).data!;
    expect(error, contains('Year must be a whole number'));
    expect(error, contains('Month must be between 1 and 12'));
    expect(error, contains('Market is missing'));
  });

  testWidgets('a valid request shows the price', (tester) async {
    final mock = MockClient((request) async => http.Response(
          '{"predicted_price_rwf": 804.0, "unit": "KG", "commodity": "Beans (dry)", "market": "Nyabugogo", '
          '"district": "Nyarugenge", "province": "Kigali City", "year": 2026, "month": 7, "price_type": "Retail", '
          '"typical_error_percent": 11.2, "records_for_this_market_and_product": 85, "notes": [], '
          '"model_name": "RandomForest", "model_version": 1}',
          200,
        ));
    await tester.pumpWidget(IsokoApp(api: ApiService(client: mock)));
    await tester.enterText(find.byKey(const Key('yearField')), '2026');
    await tester.enterText(find.byKey(const Key('monthField')), '7');
    await tester.enterText(find.byKey(const Key('marketField')), 'Nyabugogo');
    await tester.enterText(find.byKey(const Key('productField')), 'Beans (dry)');
    FocusManager.instance.primaryFocus?.unfocus();
    await tester.pump();
    await tester.tap(find.byKey(const Key('predictButton')));
    await tester.pumpAndSettle();
    expect(find.text('804 RWF per kg'), findsOneWidget);
  });

  testWidgets('a 422 error from the API is shown as readable text', (tester) async {
    final mock = MockClient((request) async => http.Response(
          '{"detail": [{"loc": ["body", "market"], "msg": "Value error, Unknown market \'Kigaly\'. Did you mean: Kigali?"}]}',
          422,
        ));
    await tester.pumpWidget(IsokoApp(api: ApiService(client: mock)));
    await tester.enterText(find.byKey(const Key('yearField')), '2026');
    await tester.enterText(find.byKey(const Key('monthField')), '7');
    await tester.enterText(find.byKey(const Key('marketField')), 'Kigaly');
    await tester.enterText(find.byKey(const Key('productField')), 'Maize');
    FocusManager.instance.primaryFocus?.unfocus();
    await tester.pump();
    await tester.tap(find.byKey(const Key('predictButton')));
    await tester.pumpAndSettle();
    expect(find.text("market: Unknown market 'Kigaly'. Did you mean: Kigali?"), findsOneWidget);
  });
}
