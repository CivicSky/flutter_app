import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const PersonDetectorApp());
}

class GateLogRecord {
  final DateTime timestamp;
  final String status;
  final String reason;
  final double confidence;
  final bool isValid;

  GateLogRecord({
    required this.timestamp,
    required this.status,
    required this.reason,
    required this.confidence,
    required this.isValid,
  });
}

class PersonDetectorApp extends StatelessWidget {
  const PersonDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Smart Gate Uniform Monitoring',
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

  Timer? _statusTimer;
  bool _isStreaming = false;
  Uint8List? frameBytes;

  int peopleCount = 0;
  bool loading = true;
  String status = 'Connecting to Python backend...';
  String? error;
  bool _statusBusy = false;

  bool hasInvalid = false;
  bool hasValid = false;
  String? activeReason;
  double activeConfidence = 0.0;
  bool soundEnabled = true;

  DateTime? _lastBeepTime;
  String? _lastLoggedStatus;
  DateTime _lastLoggedTime = DateTime.fromMillisecondsSinceEpoch(0);

  final List<GateLogRecord> _logHistory = [];
  String _selectedFilter = 'All'; // 'All', 'Valid', 'Invalid'

  @override
  void initState() {
    super.initState();
    _startVideoStream();
    _pollStatus();
    _statusTimer = Timer.periodic(
      const Duration(milliseconds: 300),
      (_) => _pollStatus(),
    );
  }

  void _triggerBeep() {
    SystemSound.play(SystemSoundType.alert);
    http.post(Uri.parse('$baseUrl/beep')).ignore();
  }

  /// High-performance video stream loop (~30 FPS)
  void _startVideoStream() async {
    _isStreaming = true;
    final client = http.Client();

    while (_isStreaming && mounted) {
      try {
        final response = await client
            .get(Uri.parse('$baseUrl/frame'))
            .timeout(const Duration(milliseconds: 1500));

        if (response.statusCode == 200 && mounted) {
          setState(() {
            frameBytes = response.bodyBytes;
            loading = false;
          });
        }
      } catch (_) {
        await Future.delayed(const Duration(milliseconds: 80));
      }

      // Smooth 30 FPS target pacing
      await Future.delayed(const Duration(milliseconds: 30));
    }

    client.close();
  }

  /// Periodic status polling (300ms interval)
  Future<void> _pollStatus() async {
    if (_statusBusy) return;
    _statusBusy = true;

    try {
      final response = await http
          .get(Uri.parse('$baseUrl/status'))
          .timeout(const Duration(seconds: 2));

      if (response.statusCode != 200) {
        throw Exception('Server returned ${response.statusCode}');
      }

      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final count = data['count'] as int? ?? 0;
      final cameraError = data['camera_error'] as String?;
      final bool serverHasInvalid = data['has_invalid'] as bool? ?? false;
      final bool serverHasValid = data['has_valid'] as bool? ?? false;
      final detections = (data['detections'] as List<dynamic>? ?? [])
          .map((e) => e as Map<String, dynamic>)
          .toList();

      final now = DateTime.now();

      // Handle Invalid detections & logging
      if (serverHasInvalid) {
        final invalidDet = detections.firstWhere(
          (d) => (d['class_name'] as String? ?? '').toLowerCase() == 'invalid',
          orElse: () => <String, dynamic>{},
        );

        final reason = (invalidDet['reason'] as String?) ??
            'Non-compliant attire: Missing or improper school uniform';
        final conf = (invalidDet['confidence'] as num?)?.toDouble() ?? 0.0;

        activeReason = reason;
        activeConfidence = conf;

        // Beep alarm
        if (soundEnabled &&
            (_lastBeepTime == null ||
                now.difference(_lastBeepTime!) > const Duration(seconds: 2))) {
          _lastBeepTime = now;
          _triggerBeep();
        }

        // Add to log (debounced)
        if (_lastLoggedStatus != 'Invalid' ||
            now.difference(_lastLoggedTime) > const Duration(seconds: 3)) {
          _lastLoggedStatus = 'Invalid';
          _lastLoggedTime = now;
          _logHistory.insert(
            0,
            GateLogRecord(
              timestamp: now,
              status: 'Invalid',
              reason: reason,
              confidence: conf,
              isValid: false,
            ),
          );
          if (_logHistory.length > 50) _logHistory.removeLast();
        }
      }
      // Handle Valid detections & logging
      else if (serverHasValid) {
        final validDet = detections.firstWhere(
          (d) => (d['class_name'] as String? ?? '').toLowerCase() == 'valid',
          orElse: () => <String, dynamic>{},
        );

        final reason = (validDet['reason'] as String?) ??
            'Prescribed uniform verified';
        final conf = (validDet['confidence'] as num?)?.toDouble() ?? 0.0;

        activeReason = reason;
        activeConfidence = conf;

        // Add to log (debounced)
        if (_lastLoggedStatus != 'Valid' ||
            now.difference(_lastLoggedTime) > const Duration(seconds: 3)) {
          _lastLoggedStatus = 'Valid';
          _lastLoggedTime = now;
          _logHistory.insert(
            0,
            GateLogRecord(
              timestamp: now,
              status: 'Valid',
              reason: reason,
              confidence: conf,
              isValid: true,
            ),
          );
          if (_logHistory.length > 50) _logHistory.removeLast();
        }
      } else {
        // No person detected in frame
        activeReason = null;
        _lastLoggedStatus = null;
      }

      if (!mounted) return;

      setState(() {
        peopleCount = count;
        error = cameraError;
        hasInvalid = serverHasInvalid;
        hasValid = serverHasValid;

        if (cameraError != null) {
          status = 'Camera unavailable';
        } else if (count == 0) {
          status = 'Gate Standby - Ready for Entrants';
        } else if (serverHasInvalid) {
          status = 'INVALID UNIFORM DETECTED';
        } else if (serverHasValid) {
          status = 'VALID - Uniform Verified';
        } else {
          status = 'Person Detected ($count)';
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        error = 'Cannot connect to Python server: $e';
        status = 'Backend disconnected';
      });
    } finally {
      _statusBusy = false;
    }
  }

  @override
  void dispose() {
    _isStreaming = false;
    _statusTimer?.cancel();
    super.dispose();
  }

  String _formatTime(DateTime dt) {
    final h = dt.hour.toString().padLeft(2, '0');
    final m = dt.minute.toString().padLeft(2, '0');
    final s = dt.second.toString().padLeft(2, '0');
    return '$h:$m:$s';
  }

  List<GateLogRecord> get _filteredLogs {
    if (_selectedFilter == 'Valid') {
      return _logHistory.where((log) => log.isValid).toList();
    } else if (_selectedFilter == 'Invalid') {
      return _logHistory.where((log) => !log.isValid).toList();
    }
    return _logHistory;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Smart Gate Uniform Monitoring'),
        centerTitle: true,
        actions: [
          IconButton(
            tooltip: soundEnabled ? 'Mute Alert Sound' : 'Enable Alert Sound',
            icon: Icon(
              soundEnabled ? Icons.volume_up : Icons.volume_off,
              color: soundEnabled ? Colors.tealAccent : Colors.grey,
            ),
            onPressed: () {
              setState(() {
                soundEnabled = !soundEnabled;
              });
            },
          ),
          IconButton(
            tooltip: 'Test Beep Sound',
            icon: const Icon(Icons.notifications_active_outlined),
            onPressed: _triggerBeep,
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final isWide = constraints.maxWidth >= 860;

          if (isWide) {
            return Padding(
              padding: const EdgeInsets.all(16.0),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    flex: 3,
                    child: _buildCameraSection(),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    flex: 2,
                    child: _buildSideNotificationPanel(),
                  ),
                ],
              ),
            );
          }

          return SingleChildScrollView(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: [
                _buildCameraSection(),
                const SizedBox(height: 16),
                _buildSideNotificationPanel(),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildCameraSection() {
    final Color statusColor;
    final IconData statusIcon;

    if (error != null) {
      statusColor = Colors.orangeAccent;
      statusIcon = Icons.warning;
    } else if (hasInvalid) {
      statusColor = Colors.redAccent;
      statusIcon = Icons.gpp_bad;
    } else if (hasValid) {
      statusColor = Colors.greenAccent;
      statusIcon = Icons.verified_user;
    } else {
      statusColor = Colors.blueAccent;
      statusIcon = Icons.people_outline;
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Card(
          elevation: 2,
          child: ListTile(
            leading: Icon(
              statusIcon,
              size: 40,
              color: statusColor,
            ),
            title: Text(
              status,
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: statusColor,
              ),
            ),
            subtitle: Text('People detected: $peopleCount  •  Smooth 30 FPS Stream'),
            trailing: Icon(
              Icons.circle,
              color: statusColor,
              size: 14,
            ),
          ),
        ),
        const SizedBox(height: 12),
        Container(
          height: 440,
          width: double.infinity,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: Colors.black,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: hasInvalid ? Colors.redAccent.withValues(alpha: 0.8) : Colors.transparent,
              width: 2,
            ),
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
        if (error != null) ...[
          const SizedBox(height: 8),
          Text(error!, style: const TextStyle(color: Colors.redAccent)),
        ],
      ],
    );
  }

  Widget _buildSideNotificationPanel() {
    final validLogsCount = _logHistory.where((l) => l.isValid).length;
    final invalidLogsCount = _logHistory.where((l) => !l.isValid).length;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Live Status Banner
        if (hasInvalid)
          Card(
            color: Colors.red.shade900.withValues(alpha: 0.4),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(14),
              side: const BorderSide(color: Colors.redAccent, width: 1.5),
            ),
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.warning_rounded, color: Colors.redAccent, size: 28),
                      const SizedBox(width: 8),
                      const Expanded(
                        child: Text(
                          'VIOLATION DETECTED',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Colors.redAccent,
                            letterSpacing: 0.5,
                          ),
                        ),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: Colors.redAccent,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: const Text(
                          'ALERT',
                          style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const Divider(color: Colors.redAccent, height: 20),
                  const Text(
                    'Why this outfit is invalid:',
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 13,
                      color: Colors.white70,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    activeReason ??
                        'Non-compliant attire: Missing or improper school uniform',
                    style: const TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'Confidence: ${(activeConfidence * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(color: Colors.white70, fontSize: 12),
                      ),
                      Row(
                        children: [
                          Icon(
                            soundEnabled ? Icons.volume_up : Icons.volume_off,
                            size: 14,
                            color: Colors.white70,
                          ),
                          const SizedBox(width: 4),
                          Text(
                            soundEnabled ? 'Beep Triggered' : 'Sound Muted',
                            style: const TextStyle(color: Colors.white70, fontSize: 12),
                          ),
                        ],
                      ),
                    ],
                  ),
                ],
              ),
            ),
          )
        else if (hasValid)
          Card(
            color: Colors.green.shade900.withValues(alpha: 0.35),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(14),
              side: const BorderSide(color: Colors.greenAccent, width: 1.5),
            ),
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Row(
                children: [
                  const Icon(Icons.check_circle_outline, color: Colors.greenAccent, size: 28),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'UNIFORM COMPLIANT',
                          style: TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.bold,
                            color: Colors.greenAccent,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          activeReason ?? 'Prescribed uniform verified. Entrant cleared.',
                          style: const TextStyle(fontSize: 13, color: Colors.white70),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Confidence: ${(activeConfidence * 100).toStringAsFixed(1)}%',
                          style: const TextStyle(fontSize: 11, color: Colors.greenAccent),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          )
        else
          Card(
            color: Colors.grey.shade900.withValues(alpha: 0.5),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(14),
              side: BorderSide(color: Colors.grey.shade700, width: 1),
            ),
            child: const Padding(
              padding: EdgeInsets.all(16.0),
              child: Row(
                children: [
                  Icon(Icons.person_search_outlined, color: Colors.blueAccent, size: 28),
                  SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Gate Monitoring Standby',
                          style: TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                        SizedBox(height: 2),
                        Text(
                          'Awaiting students at the gate for uniform verification.',
                          style: TextStyle(fontSize: 13, color: Colors.white60),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),

        const SizedBox(height: 16),

        // Detection Logs (Valid & Invalid)
        Card(
          child: Padding(
            padding: const EdgeInsets.all(14.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Row(
                      children: [
                        Icon(Icons.history, size: 20, color: Colors.tealAccent),
                        SizedBox(width: 8),
                        Text(
                          'Verification Logs',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 15,
                          ),
                        ),
                      ],
                    ),
                    if (_logHistory.isNotEmpty)
                      TextButton(
                        onPressed: () {
                          setState(() {
                            _logHistory.clear();
                          });
                        },
                        child: const Text('Clear', style: TextStyle(fontSize: 12)),
                      ),
                  ],
                ),
                const SizedBox(height: 8),

                // Filter chips
                Wrap(
                  spacing: 6,
                  children: [
                    ChoiceChip(
                      label: Text('All (${_logHistory.length})', style: const TextStyle(fontSize: 11)),
                      selected: _selectedFilter == 'All',
                      onSelected: (sel) {
                        if (sel) setState(() => _selectedFilter = 'All');
                      },
                    ),
                    ChoiceChip(
                      label: Text('Valid ($validLogsCount)', style: const TextStyle(fontSize: 11)),
                      selected: _selectedFilter == 'Valid',
                      onSelected: (sel) {
                        if (sel) setState(() => _selectedFilter = 'Valid');
                      },
                    ),
                    ChoiceChip(
                      label: Text('Invalid ($invalidLogsCount)', style: const TextStyle(fontSize: 11)),
                      selected: _selectedFilter == 'Invalid',
                      onSelected: (sel) {
                        if (sel) setState(() => _selectedFilter = 'Invalid');
                      },
                    ),
                  ],
                ),
                const Divider(),

                if (_filteredLogs.isEmpty)
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 24.0),
                    child: Center(
                      child: Text(
                        'No detection logs recorded yet.',
                        style: TextStyle(color: Colors.white54, fontSize: 13),
                      ),
                    ),
                  )
                else
                  ConstrainedBox(
                    constraints: const BoxConstraints(maxHeight: 280),
                    child: ListView.separated(
                      shrinkWrap: true,
                      itemCount: _filteredLogs.length,
                      separatorBuilder: (_, _) => const Divider(height: 1),
                      itemBuilder: (context, index) {
                        final item = _filteredLogs[index];
                        final isItemValid = item.isValid;

                        return ListTile(
                          dense: true,
                          contentPadding: EdgeInsets.zero,
                          leading: Icon(
                            isItemValid ? Icons.check_circle_outline : Icons.error_outline,
                            color: isItemValid ? Colors.greenAccent : Colors.redAccent,
                            size: 20,
                          ),
                          title: Text(
                            item.reason,
                            style: const TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                          subtitle: Text(
                            'Confidence: ${(item.confidence * 100).toStringAsFixed(1)}%',
                            style: const TextStyle(fontSize: 11, color: Colors.white60),
                          ),
                          trailing: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                            decoration: BoxDecoration(
                              color: isItemValid
                                  ? Colors.green.shade900.withValues(alpha: 0.5)
                                  : Colors.red.shade900.withValues(alpha: 0.5),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Text(
                              '${item.status.toUpperCase()}  ${_formatTime(item.timestamp)}',
                              style: TextStyle(
                                fontSize: 10,
                                fontWeight: FontWeight.bold,
                                color: isItemValid ? Colors.greenAccent : Colors.redAccent,
                              ),
                            ),
                          ),
                        );
                      },
                    ),
                  ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
