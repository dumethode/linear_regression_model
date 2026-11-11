import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

/// Public address of the API on Render. Change it here if your Render service has another name.
const String kApiBaseUrl = 'https://isoko-price-api.onrender.com';

/// What the API sends back for a successful prediction.
class PriceResult {
  PriceResult({
    required this.price,
    required this.unit,
    required this.market,
    required this.commodity,
    required this.district,
    required this.province,
    required this.typicalErrorPercent,
    required this.notes,
  });

  final double price;
  final String unit;
  final String market;
  final String commodity;
  final String? district;
  final String? province;
  final double typicalErrorPercent;
  final List<String> notes;

  factory PriceResult.fromJson(Map<String, dynamic> json) {
    return PriceResult(
      price: (json['predicted_price_rwf'] as num).toDouble(),
      unit: json['unit'] as String,
      market: json['market'] as String,
      commodity: json['commodity'] as String,
      district: json['district'] as String?,
      province: json['province'] as String?,
      typicalErrorPercent: (json['typical_error_percent'] as num).toDouble(),
      notes: (json['notes'] as List<dynamic>).map((e) => e.toString()).toList(),
    );
  }
}

/// Error with a message that can be shown to the user as it is.
class ApiException implements Exception {
  ApiException(this.message);
  final String message;

  @override
  String toString() => message;
}

class ApiService {
  ApiService({http.Client? client}) : _client = client ?? http.Client();

  final http.Client _client;

  Future<PriceResult> predict({
    required int year,
    required int month,
    required String market,
    required String commodity,
    required String priceType,
  }) async {
    final uri = Uri.parse('$kApiBaseUrl/predict');
    final body = jsonEncode({
      'year': year,
      'month': month,
      'market': market,
      'commodity': commodity,
      'price_type': priceType,
    });

    http.Response response;
    try {
      response = await _client
          .post(uri, headers: {'Content-Type': 'application/json'}, body: body)
          // A free Render server can take up to a minute to wake up.
          .timeout(const Duration(seconds: 90));
    } on TimeoutException {
      throw ApiException('The server did not answer in time. It may be waking up, please try again.');
    } on http.ClientException {
      throw ApiException('Could not reach the server. Check your internet connection and try again.');
    }

    if (response.statusCode == 200) {
      return PriceResult.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
    }
    if (response.statusCode == 422) {
      throw ApiException(_readValidationErrors(response.body));
    }
    throw ApiException('Server error (${response.statusCode}). Please try again later.');
  }

  /// Turns FastAPI's 422 error list into short lines such as "market: Unknown market 'Kigaly'. Did you mean: Kigali?".
  String _readValidationErrors(String body) {
    try {
      final decoded = jsonDecode(body) as Map<String, dynamic>;
      final details = decoded['detail'];
      if (details is List) {
        final lines = details.map((d) {
          final item = d as Map<String, dynamic>;
          final loc = (item['loc'] as List<dynamic>).last.toString().replaceAll('_', ' ');
          final msg = item['msg'].toString().replaceFirst('Value error, ', '');
          return '$loc: $msg';
        });
        return lines.join('\n');
      }
      return details.toString();
    } catch (_) {
      return 'The server did not accept these values.';
    }
  }
}
