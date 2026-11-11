import 'package:flutter/material.dart';

import 'api_service.dart';
import 'options.dart';

void main() {
  runApp(const IsokoApp());
}

class IsokoApp extends StatelessWidget {
  const IsokoApp({super.key, this.api});

  /// Can be replaced in tests.
  final ApiService? api;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Isoko Price Check',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: const Color(0xFF2E7D32), useMaterial3: true),
      home: PredictionPage(api: api ?? ApiService()),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key, required this.api});

  final ApiService api;

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _year = TextEditingController();
  final _month = TextEditingController();
  final _market = TextEditingController();
  final _product = TextEditingController();
  final _priceType = TextEditingController(text: 'Retail');
  final _marketFocus = FocusNode();
  final _productFocus = FocusNode();
  final _priceTypeFocus = FocusNode();

  bool _loading = false;
  PriceResult? _result;
  String? _error;

  @override
  void dispose() {
    for (final c in [_year, _month, _market, _product, _priceType]) {
      c.dispose();
    }
    for (final f in [_marketFocus, _productFocus, _priceTypeFocus]) {
      f.dispose();
    }
    super.dispose();
  }

  /// Checks the inputs on the phone first, so simple mistakes get a clear message without a server call.
  List<String> _checkInputs() {
    final problems = <String>[];

    void checkWholeNumber(String label, String text, int min, int max) {
      if (text.trim().isEmpty) {
        problems.add('$label is missing.');
        return;
      }
      final value = int.tryParse(text.trim());
      if (value == null) {
        problems.add('$label must be a whole number, for example $min.');
      } else if (value < min || value > max) {
        problems.add('$label must be between $min and $max.');
      }
    }

    checkWholeNumber('Year', _year.text, kMinYear, kMaxYear);
    checkWholeNumber('Month', _month.text, 1, 12);
    if (_market.text.trim().isEmpty) problems.add('Market is missing.');
    if (_product.text.trim().isEmpty) problems.add('Product is missing.');
    if (_priceType.text.trim().isEmpty) problems.add('Price type is missing (Retail or Wholesale).');
    return problems;
  }

  Future<void> _predict() async {
    FocusScope.of(context).unfocus();
    final problems = _checkInputs();
    if (problems.isNotEmpty) {
      setState(() {
        _result = null;
        _error = problems.join('\n');
      });
      return;
    }

    setState(() {
      _loading = true;
      _result = null;
      _error = null;
    });
    try {
      final result = await widget.api.predict(
        year: int.parse(_year.text.trim()),
        month: int.parse(_month.text.trim()),
        market: _market.text.trim(),
        commodity: _product.text.trim(),
        priceType: _priceType.text.trim(),
      );
      if (!mounted) return;
      setState(() => _result = result);
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() => _error = e.message);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = 'Something went wrong: $e');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Isoko Price Check'),
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text(
                'Find the usual market price of a farm product in Rwanda before you sell.',
                style: TextStyle(fontSize: 15),
              ),
              const SizedBox(height: 16),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: _numberField(_year, 'Year', '$kMinYear to $kMaxYear', const Key('yearField')),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _numberField(_month, 'Month', '1 to 12', const Key('monthField')),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              _suggestField(_market, _marketFocus, 'Market', 'e.g. Nyabugogo', kMarkets, const Key('marketField')),
              const SizedBox(height: 12),
              _suggestField(_product, _productFocus, 'Product', 'e.g. Beans (dry)', kProducts, const Key('productField')),
              const SizedBox(height: 12),
              _suggestField(_priceType, _priceTypeFocus, 'Price type', 'Retail or Wholesale', kPriceTypes,
                  const Key('priceTypeField')),
              const SizedBox(height: 20),
              SizedBox(
                height: 48,
                child: FilledButton(
                  key: const Key('predictButton'),
                  onPressed: _loading ? null : _predict,
                  child: const Text('Predict', style: TextStyle(fontSize: 16)),
                ),
              ),
              const SizedBox(height: 20),
              _resultArea(context),
            ],
          ),
        ),
      ),
    );
  }

  Widget _numberField(TextEditingController controller, String label, String hint, Key key) {
    return TextField(
      key: key,
      controller: controller,
      keyboardType: TextInputType.number,
      textInputAction: TextInputAction.next,
      decoration: InputDecoration(labelText: label, hintText: hint, border: const OutlineInputBorder()),
    );
  }

  /// A text field that suggests valid values while the user types. Any text can still be typed.
  Widget _suggestField(
    TextEditingController controller,
    FocusNode focusNode,
    String label,
    String hint,
    List<String> options,
    Key key,
  ) {
    return LayoutBuilder(
      builder: (context, constraints) => RawAutocomplete<String>(
        textEditingController: controller,
        focusNode: focusNode,
        optionsBuilder: (value) {
          final typed = value.text.trim().toLowerCase();
          if (typed.isEmpty) return const Iterable<String>.empty();
          return options.where((o) => o.toLowerCase().contains(typed));
        },
        fieldViewBuilder: (context, fieldController, fieldFocus, onSubmitted) => TextField(
          key: key,
          controller: fieldController,
          focusNode: fieldFocus,
          textInputAction: TextInputAction.next,
          decoration: InputDecoration(labelText: label, hintText: hint, border: const OutlineInputBorder()),
        ),
        optionsViewBuilder: (context, onSelected, matches) => Align(
          alignment: Alignment.topLeft,
          child: Material(
            elevation: 4,
            borderRadius: BorderRadius.circular(8),
            child: ConstrainedBox(
              constraints: BoxConstraints(maxHeight: 220, maxWidth: constraints.maxWidth),
              child: ListView.builder(
                padding: EdgeInsets.zero,
                shrinkWrap: true,
                itemCount: matches.length,
                itemBuilder: (context, i) {
                  final option = matches.elementAt(i);
                  return ListTile(dense: true, title: Text(option), onTap: () => onSelected(option));
                },
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _resultArea(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    Widget content;
    Color background = scheme.surfaceContainerHighest;

    if (_loading) {
      content = const Row(
        children: [
          SizedBox(width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 2.5)),
          SizedBox(width: 12),
          Expanded(child: Text('Getting the price... The first request can take up to a minute.')),
        ],
      );
    } else if (_error != null) {
      background = scheme.errorContainer;
      content = Text(
        _error!,
        key: const Key('errorText'),
        style: TextStyle(color: scheme.onErrorContainer, fontSize: 15),
      );
    } else if (_result != null) {
      final r = _result!;
      final place = [r.market, r.district, r.province].whereType<String>().join(', ');
      content = Column(
        key: const Key('resultText'),
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Estimated price', style: Theme.of(context).textTheme.labelLarge),
          const SizedBox(height: 4),
          Text(
            '${_formatRwf(r.price)} RWF per ${_unitName(r.unit)}',
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 6),
          Text('${r.commodity} at $place'),
          const SizedBox(height: 6),
          Text('This is an estimate. On past data it was off by about ${r.typicalErrorPercent.toStringAsFixed(0)}% on average.'),
          for (final note in r.notes) ...[
            const SizedBox(height: 6),
            Text('Note: $note', style: TextStyle(color: scheme.tertiary)),
          ],
        ],
      );
    } else {
      content = const Text('Fill in the five fields and tap Predict. The result will appear here.');
    }

    return Container(
      key: const Key('resultArea'),
      width: double.infinity,
      constraints: const BoxConstraints(minHeight: 90),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: background, borderRadius: BorderRadius.circular(12)),
      child: content,
    );
  }

  static String _unitName(String unit) {
    switch (unit) {
      case 'KG':
        return 'kg';
      case 'L':
        return 'litre';
      default:
        return 'piece';
    }
  }

  static String _formatRwf(double value) {
    final digits = value.round().toString();
    final buffer = StringBuffer();
    for (var i = 0; i < digits.length; i++) {
      if (i > 0 && (digits.length - i) % 3 == 0) buffer.write(',');
      buffer.write(digits[i]);
    }
    return buffer.toString();
  }
}
