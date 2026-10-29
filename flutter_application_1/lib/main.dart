import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const PersonDetectorApp());
}

class PersonDetectorApp extends StatelessWidget {
  const PersonDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Person Detector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.tealAccent,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const DetectorScreen(),
    );
  }
}

class DetectorScreen extends StatefulWidget {
  const DetectorScreen({super.key});

  @override
  State<DetectorScreen> createState() => _DetectorScreenState();
}

class _DetectorScreenState extends State<DetectorScreen> {
  static const String baseUrl = 'http://127.0.0.1:8000';

  Timer? timer;
  Uint8List? frameBytes;

  int peopleCount = 0;
  bool loading = true;
  String status = 'Connecting to Python...';
  String? error;

  @override
  void initState() {
    super.initState();
    refreshCamera();
    timer = Timer.periodic(
      const Duration(milliseconds: 500),
      (_) => refreshCamera(),
    );
  }

  Future<void> refreshCamera() async {
    if (_busy) return;
    _busy = true;

    try {
      final results = await Future.wait([
        http
            .get(Uri.parse('$baseUrl/status'))
            .timeout(const Duration(seconds: 3)),
        http
            .get(Uri.parse('$baseUrl/frame'))
            .timeout(const Duration(seconds: 3)),
      ]);

      final statusResponse = results[0];
      final frameResponse = results[1];

      if (statusResponse.statusCode != 200) {
        throw Exception('Cannot read Python server status');
      }

      final data = jsonDecode(statusResponse.body) as Map<String, dynamic>;

      if (frameResponse.statusCode == 200) {
        frameBytes = frameResponse.bodyBytes;
      }

      final count = data['count'] as int? ?? 0;
      final cameraError = data['camera_error'] as String?;

      if (!mounted) return;

      setState(() {
        peopleCount = count;
        loading = false;
        error = cameraError;

        if (cameraError != null) {
          status = 'Camera unavailable';
        } else if (frameResponse.statusCode != 200) {
          status = 'Waiting for camera frame...';
        } else {
          status = count > 0
              ? 'INVALID - Person Detected'
              : 'No Person Detected';
        }
      });
    } catch (e) {
      if (!mounted) return;

      setState(() {
        loading = false;
        error = 'Cannot connect to Python. Is the server running? $e';
        status = 'Server disconnected';
      });
    } finally {
      _busy = false;
    }
  }

  bool _busy = false;

  @override
  void dispose() {
    timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final detected = peopleCount > 0 && error == null;

    return Scaffold(
      appBar: AppBar(title: const Text('Person Detector'), centerTitle: true),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 1000),
          child: ListView(
            padding: const EdgeInsets.all(20),
            children: [
              Card(
                child: ListTile(
                  leading: Icon(
                    detected ? Icons.person : Icons.people_outline,
                    size: 42,
                    color: detected ? Colors.redAccent : Colors.blueAccent,
                  ),
                  title: Text(status),
                  subtitle: Text('People detected: $peopleCount'),
                  trailing: Icon(
                    Icons.circle,
                    color: detected ? Colors.redAccent : Colors.greenAccent,
                    size: 12,
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Container(
                height: 420,
                width: double.infinity,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: Colors.black,
                  borderRadius: BorderRadius.circular(16),
                ),
                clipBehavior: Clip.antiAlias,
                child: frameBytes != null
                    ? Image.memory(
                        frameBytes!,
                        fit: BoxFit.contain,
                        gaplessPlayback: true,
                        filterQuality: FilterQuality.low,
                      )
                    : loading
                    ? const CircularProgressIndicator()
                    : const Text('Waiting for webcam image...'),
              ),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: refreshCamera,
                icon: const Icon(Icons.refresh),
                label: const Text('Refresh Detection'),
              ),
              if (error != null) ...[
                const SizedBox(height: 12),
                Text(error!, style: const TextStyle(color: Colors.redAccent)),
              ],
              const SizedBox(height: 12),
              const Text(
                'Live webcam detection powered by OpenCV and YOLO. '
                'The image and detection count update automatically.',
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
