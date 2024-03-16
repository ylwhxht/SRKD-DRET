
History
-------


10.1.0 (2018-01-04)
+++++++++++++++++++

- Ability to only track folders containing a .wakatime-project file using new
  include_only_with_project_file argument and config option.


10.0.5 (2017-11-24)
+++++++++++++++++++

- Fix bug that caused heartbeats to be cached locally instead of sent to API.


10.0.4 (2017-11-23)
+++++++++++++++++++

- Improve Java dependency detection.
- Skip null or missing heartbeats from extra heartbeats argument.


10.0.3 (2017-11-22)
+++++++++++++++++++

- Support saving unicode heartbeats when working offline.
  `#112 <https://github.com/wakatime/wakatime/issues/112>`_


10.0.2 (2017-11-15)
+++++++++++++++++++

- Limit bulk syncing to 5 heartbeats per request.
  `#109 <https://github.com/wakatime/wakatime/issues/109>`_


10.0.1 (2017-11-09)
+++++++++++++++++++

- Parse array of results from bulk heartbeats endpoint, only saving heartbeats
  to local offline cache when they were not accepted by the api.


10.0.0 (2017-11-08)
+++++++++++++++++++

- Upload multiple heartbeats to bulk endpoint for improved network performance.
  `#107 <https://github.com/wakatime/wakatime/issues/107>`_


9.0.1 (2017-11-07)
++++++++++++++++++

- Fix bug causing 401 response when hidefilenames is enabled.
  `#106 <https://github.com/wakatime/wakatime/issues/106>`_


9.0.0 (2017-11-04)
++++++++++++++++++

- Detect project and branch names from git submodules.
  `#105 <https://github.com/wakatime/wakatime/issues/105>`_


8.0.5 (2017-10-29)
++++++++++++++++++

- Use WAKATIME_HOME env variable for offline and session caching.
  `#102 <https://github.com/wakatime/wakatime/issues/102>`_


8.0.4 (2017-10-23)
++++++++++++++++++

- Allow passing string arguments wrapped in extra quotes for plugins which
  cannot properly escape spaces in arguments.
- Upgrade pytz to v2017.2.
- Upgrade requests to v2.18.4.
- Upgrade tzlocal to v1.4.


8.0.3 (2017-06-08)
++++++++++++++++++

- Improve Matlab language detection.


8.0.2 (2017-05-24)
++++++++++++++++++

- Only treat proxy string as NTLM proxy after unable to connect with HTTPS and
  SOCKS proxy.
- Support running automated tests on Linux, OS X, and Windows.


8.0.1 (2017-05-05)
++++++++++++++++++

- Ability to disable SSL cert verification.
  `#90 <https://github.com/wakatime/wakatime/issues/90>`_
- Disable line count stats for files larger than 2MB to improve performance.
- Print error saying Python needs upgrading when requests can't be imported.


8.0.0 (2017-03-16)
++++++++++++++++++

- Config file not needed when passing api key via command line.
- Allow colons in [projectmap] config section.
  `#83 <https://github.com/wakatime/wakatime/issues/83>`_
- When unable to detect language and debug mode turned on, log any tracebacks.
- Increase priority of F# and TypeScript languages.


7.0.4 (2017-03-01)
++++++++++++++++++

- Add six library to satisfy missing dependency from ntlm-auth.


7.0.3 (2017-02-26)
++++++++++++++++++

- Ability to prioritize common languages over uncommon lanuages.
  `#81 <https://github.com/wakatime/wakatime/issues/81>`_


7.0.2 (2017-02-20)
++++++++++++++++++

- Rename alternate_language to language in extra heartbeats json.


7.0.1 (2017-02-20)
++++++++++++++++++

- Rename --alternate-language cli argument to --language.


7.0.0 (2017-02-20)
++++++++++++++++++

- Existing --alternate-language cli argument now overwrites auto-detected
  language. Previously, was only used when unable to auto-detect language.
- Give TypeScript higher priority than TypoScript.


6.3.0 (2017-02-16)
++++++++++++++++++

- Support for Python 3.6.


6.2.3 (2017-02-15)
++++++++++++++++++

- Support NTLM proxy format like domain\\user:pass.
  `#23 <https://github.com/wakatime/wakatime/issues/23>`_
- Upgrade pytz to v2016.10.
- Upgrade requests to v2.13.0.
- Upgrade pysocks to v1.6.6.


6.2.2 (2017-02-13)
++++++++++++++++++

- Upgrade pygments library to v2.2.0 for improved language detection.


6.2.1 (2017-02-08)
++++++++++++++++++

- Allow boolean or list of regex patterns for hidefilenames config setting.


6.2.0 (2016-10-24)
++++++++++++++++++

- New WAKATIME_HOME env variable for setting path to config and log files.
  `#67 <https://github.com/wakatime/wakatime/issues/67>`_
- Improve debug warning message from unsupported dependency parsers.
  `#65 <https://github.com/wakatime/wakatime/issues/65>`_


6.1.0 (2016-10-11)
++++++++++++++++++

- Exit with status code 104 when api key is missing or invalid. Exit with
  status code 103 when config file missing or invalid.


6.0.9 (2016-09-13)
++++++++++++++++++

- Force file path to use system path separator.
- Handle exception from Python system library read permission problem.


6.0.8 (2016-09-02)
++++++++++++++++++

- Prevent encoding errors when logging files with special characters.
- Upgrade pytz to v2016.6.1.
- Upgrade requests to v2.11.1.
- Upgrade simplejson to v3.8.2.
- Upgrade tzlocal to v1.2.2.


6.0.7 (2016-07-06)
++++++++++++++++++

- Handle unknown exceptions from requests library by deleting cached session
  object because it could be from a previous conflicting version.
- New hostname setting in config file to set machine hostname. Hostname
  argument takes priority over hostname from config file.
- Prevent logging unrelated exception when logging tracebacks.


6.0.6 (2016-06-17)
++++++++++++++++++

- Use correct namespace for pygments.lexers.ClassNotFound exception so it is
  caught when dependency detection not available for a language.


6.0.5 (2016-06-13)
++++++++++++++++++

- Upgrade pygments to v2.1.3 for better language coverage.


6.0.4 (2016-06-08)
++++++++++++++++++

- Upgrade urllib3 to master version to fix bug causing unhandled retry
  exceptions.
- Prevent tracking git branch with detached head.


6.0.3 (2016-05-21)
++++++++++++++++++

- Upgrade requests dependency to v2.10.0.
- Support for SOCKS proxies.


6.0.2 (2016-05-16)
++++++++++++++++++

- Prevent popup on Mac when xcode-tools is not installed.


6.0.1 (2016-04-28)
++++++++++++++++++

- Fix bug which prevented plugin from being sent with extra heartbeats.


6.0.0 (2016-04-28)
++++++++++++++++++

- Increase default network timeout to 60 seconds when sending heartbeats to
  the api.
- New --extra-heartbeats command line argument for sending a JSON array of
  extra queued heartbeats to STDIN.
- Change --entitytype command line argument to --entity-type.
- No longer allowing --entity-type of url.


5.0.1 (2016-04-19)
++++++++++++++++++

- Support passing an alternate language to cli to be used when a language can
  not be guessed from the code file.


5.0.0 (2016-04-18)
++++++++++++++++++

- Support regex patterns in projectmap config section for renaming projects.
- Upgrade pytz to v2016.3.
- Upgrade tzlocal to v1.2.2.


4.1.13 (2016-03-06)
++++++++++++++++++

- Encode TimeZone as utf-8 before adding to headers.


4.1.12 (2016-03-06)
++++++++++++++++++

- Encode hostname as utf-8 basestring before adding to X-Machine-Name header.


4.1.11 (2016-03-06)
++++++++++++++++++

- Encode machine hostname as Unicode when adding to X-Machine-Name header.


4.1.10 (2016-01-11)
++++++++++++++++++

- Accept 201 or 202 response codes as success from api.
- Upgrade requests package to v2.9.1.


4.1.9 (2016-01-06)
++++++++++++++++++

- Improve C# dependency detection.
- Correctly log exception tracebacks.
- Log all unknown exceptions to wakatime.log file.
- Disable urllib3 SSL warning from every request.
- Detect dependencies from golang files.
- Use api.wakatime.com for sending heartbeats.


4.1.8 (2015-09-29)
++++++++++++++++++

- Fix bug in guess_language function.
- Improve dependency detection.


4.1.7 (2015-09-16)
++++++++++++++++++

- Default request timeout of 30 seconds.
- New --timeout command line argument to change request timeout in seconds.


4.1.6 (2015-09-06)
++++++++++++++++++

- Allow passing command line arguments using sys.argv.


4.1.5 (2015-09-06)
++++++++++++++++++

- Fix entry point for pypi distribution.


4.1.4 (2015-08-29)
++++++++++++++++++

- New --entity and --entitytype command line arguments.


4.1.3 (2015-08-28)
++++++++++++++++++

- Fix local session caching.


4.1.2 (2015-08-25)
++++++++++++++++++

- Fix bug in offline caching which prevented heartbeats from being cleaned up.


4.1.1 (2015-08-25)
++++++++++++++++++

- Send hostname in X-Machine-Name header.
- Catch exceptions from pygments.modeline.get_filetype_from_buffer.
- Upgrade requests package to v2.7.0.
- Handle non-ASCII characters in import path on Windows, won't fix for Python2.
- Upgrade argparse to v1.3.0.
- Move language translations to api server.
- Move extension rules to api server.
- Detect correct header file language based on presence of .cpp or .c files
  named the same as the .h file.


4.1.0 (2015-06-29)
++++++++++++++++++

- Correct priority for project detection.


4.0.16 (2015-06-23)
++++++++++++++++++

- Fix offline logging.
- Limit language detection to known file extensions, unless file contents has
  a vim modeline.


4.0.15 (2015-06-11)
++++++++++++++++++

- Guess language using multiple methods, then use most accurate guess.
- Use entity and type for new heartbeats api resource schema.


4.0.14 (2015-05-31)
++++++++++++++++++

- Correctly log message from py.warnings module.


4.0.13 (2015-05-16)
++++++++++++++++++

- Fix bug with auto detecting project name.


4.0.12 (2015-05-15)
++++++++++++++++++

- Correctly display caller and lineno in log file when debug is true.
- Project passed with --project argument will always be used.
- New --alternate-project argument.


4.0.11 (2015-05-12)
++++++++++++++++++

- Reuse SSL connection across multiple processes for improved performance.


4.0.10 (2015-05-06)
++++++++++++++++++

- New --cursorpos argument for passing index of cursor within the file
  contents.


4.0.9 (2015-05-06)
++++++++++++++++++

- New --lineno argument for passing line number of cursor at time of heartbeat.
- Format py.warnings log messages same as other log messages.
- Include package namespace and line number in logger output.


4.0.8 (2015-04-04)
++++++++++++++++++

- Added api_url config option and --apiurl cli argument for customizing api
  url.


4.0.7 (2015-04-02)
++++++++++++++++++

- Capture warnings in log file.


4.0.6 (2015-03-31)
++++++++++++++++++

- Add requests.packages directory to sys.path.


4.0.5 (2015-03-31)
++++++++++++++++++

- Update requests package to v2.0.6.
- Update simplejson to v3.6.5.


4.0.4 (2015-03-09)
++++++++++++++++++

- Add back --ignore argument for backwards compatibility.


4.0.3 (2015-03-09)
++++++++++++++++++

- Refactor module structure.


4.0.2 (2015-03-07)
++++++++++++++++++

- Include cacert.pem file in pypi distribution for SSL with requests package.


4.0.1 (2015-03-03)
++++++++++++++++++

- Upgrade requests library to v2.5.3 to fix SSL problem on CentOS.
- New options for excluding and including directories.


4.0.0 (2015-02-12)
++++++++++++++++++

- Use requests library instead of urllib2, so api SSL cert is verified.
- New --notfile argument to support logging time without a real file.
- New --proxy argument for https proxy support.


3.0.5 (2015-01-13)
++++++++++++++++++

- Ignore errors from malformed markup (too many closing tags).


3.0.4 (2015-01-06)
++++++++++++++++++

- Remove unused dependency, which is missing in some python environments.


3.0.3 (2014-12-25)
++++++++++++++++++

- Detect JavaScript frameworks from script tags in Html template files.


3.0.2 (2014-12-25)
++++++++++++++++++

- Detect frameworks from JavaScript and JSON files.


3.0.1 (2014-12-23)
++++++++++++++++++

- Handle unknown language when parsing dependencies.


3.0.0 (2014-12-23)
++++++++++++++++++

- Detect libraries and frameworks for C++, Java, .NET, PHP, and Python files.


2.1.11 (2014-12-22)
+++++++++++++++++++

- Fix offline logging when response from api is None.


2.1.10 (2014-12-15)
+++++++++++++++++++

- Prevent queuing offline heartbeats which will never be valid (400 errors).


2.1.9 (2014-12-05)
++++++++++++++++++

- Fix bug preventing offline heartbeats from being purged after uploaded.


2.1.8 (2014-12-04)
++++++++++++++++++

- Fix UnicodeDecodeError when building user agent string.
- Handle case where response is None.


2.1.7 (2014-11-30)
++++++++++++++++++

- Upgrade pygments to v2.0.1.
- Always log an error when api key is incorrect.


2.1.6 (2014-11-18)
++++++++++++++++++

- Fix list index error when detecting subversion project.


2.1.5 (2014-11-17)
++++++++++++++++++

- Catch exceptions when getting current machine time zone.


2.1.4 (2014-11-12)
++++++++++++++++++

- When Python was not compiled with https support, log an error to the log
  file.


2.1.3 (2014-11-10)
++++++++++++++++++

- Correctly detect branch name for subversion projects.


2.1.2 (2014-10-07)
++++++++++++++++++

- Still log heartbeat when something goes wrong while reading num lines in
  file.


2.1.1 (2014-09-30)
++++++++++++++++++

- Fix bug where binary file opened as utf-8.


2.1.0 (2014-09-30)
++++++++++++++++++

- Python3 compatibility changes.


2.0.8 (2014-08-29)
++++++++++++++++++

- Supress output from svn command.


2.0.7 (2014-08-27)
++++++++++++++++++

- Find svn binary location from common install directories.


2.0.6 (2014-08-07)
++++++++++++++++++

- Encode json data as str when passing to urllib.


2.0.5 (2014-07-25)
++++++++++++++++++

- Option in .wakatime.cfg to obfuscate file names.


2.0.4 (2014-07-25)
++++++++++++++++++

- Use unique logger namespace to prevent collisions in shared plugin
  environments.


2.0.3 (2014-06-18)
++++++++++++++++++

- Use project from command line arg when no revision control project is found.


2.0.2 (2014-06-09)
++++++++++++++++++

- Include python3.2 compatible versions of simplejson, pytz, and tzlocal.
- Disable offline logging when Python was not compiled with sqlite3 module.


2.0.1 (2014-05-26)
++++++++++++++++++

- Fix bug in queue preventing actions with NULL values from being purged.


2.0.0 (2014-05-25)
++++++++++++++++++

- Offline time logging using sqlite3 to queue editor events.


1.0.2 (2014-05-06)
++++++++++++++++++

- Ability to set project from command line argument.


1.0.1 (2014-03-05)
++++++++++++++++++

- Use new domain name wakatime.com.


1.0.0 (2014-02-05)
++++++++++++++++++

- Detect project name and branch name from mercurial revision control.


0.5.3 (2014-01-15)
++++++++++++++++++

- Bug fix for unicode in Python3.


0.5.2 (2014-01-14)
++++++++++++++++++

- Minor bug fix for Subversion on non-English systems.


0.5.1 (2013-12-13)
++++++++++++++++++

- Second line in .wakatime-project file now sets branch name.


0.5.0 (2013-12-13)
++++++++++++++++++

- Convert ~/.wakatime.conf to ~/.wakatime.cfg and use configparser format.
- new [projectmap] section in cfg file for naming projects based on folders.


0.4.10 (2013-11-13)
+++++++++++++++++++

- Placing .wakatime-project file in a folder will read the project's name from
  that file.


0.4.9 (2013-10-27)
++++++++++++++++++

- New config for ignoring files from regular expressions.
- Parse more options from config file (verbose, logfile, ignore).


0.4.8 (2013-10-13)
++++++++++++++++++

- Read git HEAD file to find current branch instead of running git command
  line.


0.4.7 (2013-09-30)
++++++++++++++++++

- Sending local olson timezone string in api request.


0.4.6 (2013-09-22)
++++++++++++++++++

- Sending total lines in file and language name to api.


0.4.5 (2013-09-07)
++++++++++++++++++

- Fixed relative import error by adding packages directory to sys path.


0.4.4 (2013-09-06)
++++++++++++++++++

- Using urllib2 again because of intermittent problems sending json with
  requests library.


0.4.3 (2013-09-04)
++++++++++++++++++

- Encoding json as utf-8 before making request.


0.4.2 (2013-09-04)
++++++++++++++++++

- Using requests package v1.2.3 from pypi.


0.4.1 (2013-08-25)
++++++++++++++++++

- Fix bug causing requests library to omit POST content.


0.4.0 (2013-08-15)
++++++++++++++++++

- Sending single branch instead of multiple tags.


0.3.1 (2013-08-08)
++++++++++++++++++

- Using requests module instead of urllib2 to verify SSL certs.


0.3.0 (2013-08-08)
++++++++++++++++++

- Allow importing directly from Python plugins.


0.1.1 (2013-07-07)
++++++++++++++++++

- Refactored.
- Simplified action events schema.


0.0.1 (2013-07-05)
++++++++++++++++++

- Birth.
